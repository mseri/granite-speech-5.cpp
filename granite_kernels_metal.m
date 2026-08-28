/* Metal/MPS backend. Shared buffers let GEMMs use cached weights in place. */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "granite_kernels_metal.h"


static id<MTLDevice>              g_device = nil;
static id<MTLCommandQueue>        g_queue  = nil;
static int                        g_available = 0;
static int                        g_verbose = 1;
static dispatch_once_t            g_once;
static pthread_mutex_t            g_lock = PTHREAD_MUTEX_INITIALIZER;

/* MPS compiles each matrix shape once; cache the shapes used by the encoder. */
#define MM_SLOTS 16
static struct {
    int seq, in_dim, out_dim, has_bias;
    MPSMatrixMultiplication *mm;
    MPSMatrixDescriptor *dx, *dW, *dy;
} g_mm[MM_SLOTS];
static int g_mm_count = 0;


/* Keep strong buffer references separately; registry entries are non-owning. */
typedef struct {
    void                            *base;
    size_t                           len;
    __unsafe_unretained id<MTLBuffer> buf;
} reg_entry_t;

static NSMutableArray *g_owned = nil;
static reg_entry_t    *g_reg = NULL;
static size_t          g_reg_count = 0, g_reg_cap = 0;

/* Caller must hold g_lock. */
static int reg_add(void *base, size_t len, id<MTLBuffer> buf) {
    if (g_reg_count == g_reg_cap) {
        size_t cap = g_reg_cap ? g_reg_cap * 2 : 512;
        reg_entry_t *r = realloc(g_reg, cap * sizeof(*r));
        if (!r) return 0;
        g_reg = r;
        g_reg_cap = cap;
    }
    if (!g_owned) g_owned = [NSMutableArray array];
    [g_owned addObject:buf];
    g_reg[g_reg_count].base = base;
    g_reg[g_reg_count].len  = len;
    g_reg[g_reg_count].buf  = buf;
    g_reg_count++;
    return 1;
}

/* Resolve a host pointer to its owning buffer plus byte offset. Returns nil if
 * the range [p, p+bytes) is not fully inside a registered allocation. */
static id<MTLBuffer> reg_find(const void *p, size_t bytes, size_t *offset) {
    if (!p) return nil;
    const char *c = (const char *)p;
    for (size_t i = 0; i < g_reg_count; i++) {
        const char *b = (const char *)g_reg[i].base;
        if (c >= b && c + bytes <= b + g_reg[i].len) {
            *offset = (size_t)(c - b);
            return g_reg[i].buf;
        }
    }
    return nil;
}



/* One slot per GEMM operand that might not be resident: 0=x, 1=W, 2=y. In
 * practice only x is ever staged (the front-end features); weights and
 * activations come from granite_metal_alloc and resolve in place. */
#define STAGE_SLOTS 3
static id<MTLBuffer> g_stage[STAGE_SLOTS];
static size_t        g_stage_cap[STAGE_SLOTS];

static id<MTLBuffer> stage_buf(int slot, size_t bytes) {
    if (g_stage_cap[slot] >= bytes) return g_stage[slot];

    /* Round to 4 MiB so a growing window does not reallocate every decode. */
    const size_t GRAIN = 4u << 20;
    size_t cap = (bytes + GRAIN - 1) & ~(GRAIN - 1);

    id<MTLBuffer> b = [g_device newBufferWithLength:cap
                                           options:MTLResourceStorageModeShared];
    if (!b) {
        if (g_verbose >= 1)
            fprintf(stderr, "[metal] staging allocation failed (%zu bytes)\n", cap);
        return nil;
    }
    g_stage[slot]     = b;
    g_stage_cap[slot] = cap;
    return b;
}

/* Resolve a resident operand or copy it to a staging slot. */
static id<MTLBuffer> operand(const void *p, size_t bytes, int slot, int copy_in,
                             size_t *offset) {
    id<MTLBuffer> b = reg_find(p, bytes, offset);
    /* MPS wants a naturally aligned start; every resident interior pointer we
     * hand it is row-aligned, but check rather than assume. */
    if (b && (*offset % 16) == 0) return b;

    /* Stage small unregistered operands; large ones fall back to the CPU. */
    if (bytes > (256u << 20)) return nil;

    b = stage_buf(slot, bytes);
    if (!b) return nil;
    *offset = 0;
    if (copy_in) memcpy([b contents], p, bytes);
    return b;
}


static int metal_linear(float *y, const float *x, const float *W,
                        const float *bias, int seq, int in_dim, int out_dim);

void granite_metal_set_verbose(int level) { g_verbose = level; }

void granite_metal_init(void) {
    dispatch_once(&g_once, ^{
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            if (g_verbose >= 1)
                fprintf(stderr, "[metal] no Metal device available\n");
            return;
        }
        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            if (g_verbose >= 1)
                fprintf(stderr, "[metal] failed to create command queue\n");
            g_device = nil;
            return;
        }

        g_available = 1;
        if (g_verbose >= 2)
            fprintf(stderr, "[metal] using device: %s\n",
                    [[g_device name] UTF8String]);

        /* Compile a representative GEMM before the first real layer. */
        enum { WM = 128, WK = 512, WN = 512 };
        float *a = calloc((size_t)WM * WK, sizeof(float));
        float *b = calloc((size_t)WN * WK, sizeof(float));
        float *c = calloc((size_t)WM * WN, sizeof(float));
        if (a && b && c) metal_linear(c, a, b, NULL, WM, WK, WN);
        free(a); free(b); free(c);
    });
}

void granite_metal_shutdown(void) {
    if (!g_available) return;
    pthread_mutex_lock(&g_lock);
    [g_owned removeAllObjects];
    g_owned = nil;
    free(g_reg);
    g_reg = NULL;
    g_reg_count = g_reg_cap = 0;
    for (int i = 0; i < STAGE_SLOTS; i++) { g_stage[i] = nil; g_stage_cap[i] = 0; }
    for (int i = 0; i < MM_SLOTS; i++) {
        g_mm[i].mm = nil; g_mm[i].dx = nil; g_mm[i].dW = nil; g_mm[i].dy = nil;
        g_mm[i].seq = g_mm[i].in_dim = g_mm[i].out_dim = g_mm[i].has_bias = 0;
    }
    g_mm_count = 0;
    g_queue = nil;
    g_device = nil;
    g_available = 0;
    pthread_mutex_unlock(&g_lock);
}

int granite_metal_available(void) { return g_available; }


void *granite_metal_alloc(size_t bytes) {
    granite_metal_init();
    if (!g_available || bytes == 0) return NULL;

    pthread_mutex_lock(&g_lock);
    id<MTLBuffer> b = [g_device newBufferWithLength:bytes
                                           options:MTLResourceStorageModeShared];
    void *p = b ? [b contents] : NULL;
    /* Unregistered buffers would be staged on every GEMM, so reject them. */
    if (p && !reg_add(p, bytes, b)) p = NULL;
    pthread_mutex_unlock(&g_lock);
    return p;
}

int granite_metal_dealloc(void *p) {
    if (!p || !g_available) return 0;
    pthread_mutex_lock(&g_lock);
    for (size_t i = 0; i < g_reg_count; i++) {
        if (g_reg[i].base == p) {
            [g_owned removeObjectIdenticalTo:g_reg[i].buf];  /* releases it */
            g_reg[i] = g_reg[--g_reg_count];                 /* swap-remove */
            pthread_mutex_unlock(&g_lock);
            return 1;
        }
    }
    pthread_mutex_unlock(&g_lock);
    return 0;
}


int granite_metal_should_offload(int seq, int in_dim, int out_dim) {
    if (!g_available) return 0;
    if (seq <= 1) return 0;
    /* Small matrices are faster on the CPU because dispatch overhead dominates. */
    return (long long)seq * in_dim * out_dim * 2 >= 8000000LL;
}


/* The bias-dependent beta value is fixed when MPS creates the operation. */
static int mm_slot(int seq, int in_dim, int out_dim, int has_bias) {
    for (int i = 0; i < g_mm_count; i++)
        if (g_mm[i].seq == seq && g_mm[i].in_dim == in_dim &&
            g_mm[i].out_dim == out_dim && g_mm[i].has_bias == has_bias)
            return i;
    if (g_mm_count == MM_SLOTS) g_mm_count = 0;   /* reuse an old slot */
    int i = g_mm_count++;

    g_mm[i].seq = seq; g_mm[i].in_dim = in_dim; g_mm[i].out_dim = out_dim;
    g_mm[i].has_bias = has_bias;
    g_mm[i].dx = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)seq
                                                      columns:(NSUInteger)in_dim
                                                     rowBytes:(NSUInteger)in_dim * sizeof(float)
                                                     dataType:MPSDataTypeFloat32];
    g_mm[i].dW = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)out_dim
                                                      columns:(NSUInteger)in_dim
                                                     rowBytes:(NSUInteger)in_dim * sizeof(float)
                                                     dataType:MPSDataTypeFloat32];
    g_mm[i].dy = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)seq
                                                      columns:(NSUInteger)out_dim
                                                     rowBytes:(NSUInteger)out_dim * sizeof(float)
                                                     dataType:MPSDataTypeFloat32];
    g_mm[i].mm =
        [[MPSMatrixMultiplication alloc] initWithDevice:g_device
                                         transposeLeft:NO
                                        transposeRight:YES
                                            resultRows:(NSUInteger)seq
                                         resultColumns:(NSUInteger)out_dim
                                       interiorColumns:(NSUInteger)in_dim
                                                 alpha:1.0
                                                  beta:(has_bias ? 1.0 : 0.0)];
    return i;
}

/* Linear implementation without initialization, used by the startup warm-up. */
static int metal_linear(float *y, const float *x, const float *W,
                        const float *bias, int seq, int in_dim, int out_dim) {
    const size_t sx = (size_t)seq * in_dim * sizeof(float);
    const size_t sW = (size_t)out_dim * in_dim * sizeof(float);
    const size_t sy = (size_t)seq * out_dim * sizeof(float);

    size_t ox = 0, oW = 0, oy = 0;
    id<MTLBuffer> bx = operand(x, sx, 0, /*copy_in=*/1, &ox);
    id<MTLBuffer> bW = operand(W, sW, 1, /*copy_in=*/1, &oW);
    /* The bias prefill is the only read of y. */
    id<MTLBuffer> by = operand(y, sy, 2, /*copy_in=*/0, &oy);
    if (!bx || !bW || !by) return 0;

    /* Prefill C with bias: MPS uses beta=1, and touching the pages avoids stale
     * contents when the GPU writes the result. */
    float *yd = (float *)((char *)[by contents] + oy);
    if (bias) {
        for (int i = 0; i < seq; i++)
            memcpy(yd + (size_t)i * out_dim, bias, (size_t)out_dim * sizeof(float));
    }

    int slot = mm_slot(seq, in_dim, out_dim, bias != NULL);
    MPSMatrix *mx = [[MPSMatrix alloc] initWithBuffer:bx offset:ox descriptor:g_mm[slot].dx];
    MPSMatrix *mW = [[MPSMatrix alloc] initWithBuffer:bW offset:oW descriptor:g_mm[slot].dW];
    MPSMatrix *my = [[MPSMatrix alloc] initWithBuffer:by offset:oy descriptor:g_mm[slot].dy];

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    [g_mm[slot].mm encodeToCommandBuffer:cmd leftMatrix:mx rightMatrix:mW resultMatrix:my];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (by == g_stage[2]) memcpy(y, yd, sy);
    return 1;
}

int granite_metal_linear(float *y, const float *x, const float *W,
                         const float *bias, int seq, int in_dim, int out_dim) {
    granite_metal_init();
    if (!g_available) return 0;
    return metal_linear(y, x, W, bias, seq, in_dim, out_dim);
}
