/*
 * granite_kernels_metal.m - Metal/MPS backend for Granite Speech 5.0
 *
 * See granite_kernels_metal.h for the contract. Notes on the parts that are
 * easy to get wrong:
 *
 * Buffer residency. Activations and the bf16 -> f32 weight cache are allocated
 * through granite_metal_alloc(), which hands back the `contents` pointer of a
 * MTLResourceStorageModeShared buffer and records the range in `reg`. Every
 * GEMM then resolves its operands to (buffer, offset) by range lookup, so the
 * common case moves no bytes at all. Only unregistered operands -- the
 * front-end features, and the attention distance table -- get staged.
 *
 * Fast math is disabled when compiling the shader. The regression suite
 * requires *exact* argmax agreement with reference.py, and fast-math exp()
 * is not accurate enough to guarantee that on near-ties.
 *
 * All entry points are called from the single thread that drives the encoder,
 * so the registry and the staging slots need no locking. granite_metal_alloc()
 * is the exception -- it is reachable from the weight cache under its own
 * mutex -- so it takes `g_lock`.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "granite_kernels_metal.h"

/* Threadgroup width for the attention kernel. Must equal GRANITE_DIM_HEAD and
 * be >= GRANITE_CONTEXT_SIZE; both are 128, which is also a clean 4 simdgroups. */
#define MTL_ATTN_WIDTH 128
#define MTL_ATTN_SIMDS (MTL_ATTN_WIDTH / 32)

/* ========================================================================
 * Shader source
 *
 * Embedded rather than shipped as a .metal/.metallib so the build stays a
 * plain clang invocation with no metal toolchain step. Compiles in ~30 ms at
 * init, once.
 * ======================================================================== */

static NSString *const kShaderSource = @R"METAL(
#include <metal_stdlib>
using namespace metal;

#define ATTN_WIDTH 128
#define ATTN_SIMDS 4

struct attn_params {
    uint  block_len;
    uint  heads;
    uint  dim_head;
    uint  ctx;
    uint  inner;      /* heads * dim_head, the q/k/v row stride */
    float scale;
};

/*
 * One threadgroup per (block, head, query) triple, ATTN_WIDTH threads wide.
 *
 * Phase 1: thread j scores key j -- content dot plus Shaw positional dot,
 *          mirroring the CPU kernel's two separate accumulators so the
 *          rounding matches.
 * Phase 2: thread d accumulates output dimension d over all keys.
 */
kernel void granite_block_attention(
    device       float       *out   [[buffer(0)]],
    device const float       *q     [[buffer(1)]],
    device const float       *k     [[buffer(2)]],
    device const float       *v     [[buffer(3)]],
    device const float       *rel   [[buffer(4)]],
    device const int         *dists [[buffer(5)]],
    constant     attn_params &P     [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float sq[ATTN_WIDTH];     /* the query row, read block_len times */
    threadgroup float ss[ATTN_WIDTH];     /* scores, then softmax weights */
    threadgroup float part[ATTN_SIMDS];   /* cross-simdgroup reduction */

    const uint i    = tgid % P.block_len;
    const uint h    = (tgid / P.block_len) % P.heads;
    const uint blk  =  tgid / (P.block_len * P.heads);
    const uint base = blk * P.block_len;
    const uint hoff = h * P.dim_head;

    if (tid < P.dim_head)
        sq[tid] = q[(ulong)(base + i) * P.inner + hoff + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float s = -INFINITY;
    if (tid < P.block_len) {
        device const float *kj = k + (ulong)(base + tid) * P.inner + hoff;
        device const float *rl = rel + (ulong)dists[i * P.ctx + tid] * P.dim_head;
        float dot = 0.0f, pos = 0.0f;
        for (uint d = 0; d < P.dim_head; ++d) {
            dot += sq[d] * kj[d];
            pos += sq[d] * rl[d];
        }
        s = (dot + pos) * P.scale;
    }

    /* max over the block */
    float mx = simd_max(s);
    if (lane == 0) part[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = part[0];
    for (uint g = 1; g < ATTN_SIMDS; ++g) mx = max(mx, part[g]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* exp and sum. Inactive lanes hold 0 and drop out of both reductions. */
    float e = (tid < P.block_len) ? exp(s - mx) : 0.0f;
    float sum = simd_sum(e);
    if (lane == 0) part[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = 0.0f;
    for (uint g = 0; g < ATTN_SIMDS; ++g) sum += part[g];

    if (tid < P.block_len) ss[tid] = e / sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < P.dim_head) {
        float acc = 0.0f;
        for (uint j = 0; j < P.block_len; ++j)
            acc += ss[j] * v[(ulong)(base + j) * P.inner + hoff + tid];
        out[(ulong)(base + i) * P.inner + hoff + tid] = acc;
    }
}
)METAL";

typedef struct {
    uint32_t block_len, heads, dim_head, ctx, inner;
    float    scale;
} attn_params_t;

/* ========================================================================
 * Device state
 * ======================================================================== */

static id<MTLDevice>              g_device = nil;
static id<MTLCommandQueue>        g_queue  = nil;
static id<MTLComputePipelineState> g_attn  = nil;
static int                        g_available = 0;
static int                        g_verbose = 1;
static dispatch_once_t            g_once;
static pthread_mutex_t            g_lock = PTHREAD_MUTEX_INITIALIZER;

/*
 * MPSMatrixMultiplication compiles a kernel for its shape on construction, so
 * building a fresh one per call charged that to every layer. The encoder only
 * ever uses a handful of distinct (seq, in_dim, out_dim) triples -- the frame
 * count only changes at the two subsampling blocks -- so a tiny cache removes
 * essentially all of the churn.
 */
#define MM_SLOTS 16
static struct {
    int seq, in_dim, out_dim, has_bias;
    MPSMatrixMultiplication *mm;
    MPSMatrixDescriptor *dx, *dW, *dy;
} g_mm[MM_SLOTS];
static int g_mm_count = 0;

/* ------------------------------------------------------- buffer registry --- */

/*
 * Ownership is deliberately split. `g_owned` holds the only strong references;
 * the registry array holds __unsafe_unretained aliases so the struct stays
 * trivially copyable and can be realloc'd and swap-removed as plain memory.
 * A strong field here would mean every realloc and every swap-remove had to
 * cooperate with ARC's release-the-old-value semantics, which is exactly the
 * kind of subtlety that turns into a double release later.
 */
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

/* ------------------------------------------------------------ staging pool -- */

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

/* Resolve an operand: use it in place when resident, otherwise copy it into a
 * staging slot. `copy_in` is 0 for pure outputs, whose contents are irrelevant.
 * Sets *offset and returns nil on failure. */
static id<MTLBuffer> operand(const void *p, size_t bytes, int slot, int copy_in,
                             size_t *offset) {
    id<MTLBuffer> b = reg_find(p, bytes, offset);
    /* MPS wants a naturally aligned start; every resident interior pointer we
     * hand it is row-aligned, but check rather than assume. */
    if (b && (*offset % 16) == 0) return b;

    /* Not resident. Staging is for small strays like the front-end features; a
     * large operand landing here means granite_device_alloc fell back to
     * malloc, and mirroring hundreds of MB into a second buffer is worse than
     * just running this GEMM on the CPU. The 30-minute logits buffer
     * (n_frames * 16384 floats, ~1.4 GB) is the case this guards. */
    if (bytes > (256u << 20)) return nil;

    b = stage_buf(slot, bytes);
    if (!b) return nil;
    *offset = 0;
    if (copy_in) memcpy([b contents], p, bytes);
    return b;
}

/* ------------------------------------------------- resident const uploads -- */

/* Small read-only tables (the attention distance matrix) that live in plain
 * malloc'd memory for the model's lifetime. Uploaded once, keyed by pointer. */
#define CONST_SLOTS 4
static struct { const void *src; size_t len; id<MTLBuffer> buf; } g_const[CONST_SLOTS];
static int g_const_count = 0;

static id<MTLBuffer> const_upload(const void *p, size_t bytes) {
    for (int i = 0; i < g_const_count; i++)
        if (g_const[i].src == p && g_const[i].len == bytes) return g_const[i].buf;

    id<MTLBuffer> b = [g_device newBufferWithBytes:p
                                           length:bytes
                                          options:MTLResourceStorageModeShared];
    if (!b) return nil;
    if (g_const_count < CONST_SLOTS) {
        g_const[g_const_count].src = p;
        g_const[g_const_count].len = bytes;
        g_const[g_const_count].buf = b;
        g_const_count++;
    }
    return b;
}

/* ========================================================================
 * Lifecycle
 * ======================================================================== */

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

        MTLCompileOptions *opts = [MTLCompileOptions new];
        /* Parity with reference.py requires exact argmax agreement; fast-math
         * exp() is not accurate enough to guarantee it on near-ties. */
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }

        NSError *err = nil;
        id<MTLLibrary> lib = [g_device newLibraryWithSource:kShaderSource
                                                   options:opts
                                                     error:&err];
        if (!lib) {
            if (g_verbose >= 1)
                fprintf(stderr, "[metal] shader compilation failed: %s\n",
                        [[err localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"granite_block_attention"];
        g_attn = [g_device newComputePipelineStateWithFunction:fn error:&err];
        if (!g_attn) {
            if (g_verbose >= 1)
                fprintf(stderr, "[metal] pipeline creation failed: %s\n",
                        [[err localizedDescription] UTF8String]);
            g_queue = nil;
            g_device = nil;
            return;
        }

        g_available = 1;
        if (g_verbose >= 2)
            fprintf(stderr, "[metal] using device: %s\n",
                    [[g_device name] UTF8String]);

        /* Warm the MPS GEMM so its shader JIT is not charged to the first
         * layer. Must call metal_linear directly: granite_metal_linear would
         * re-enter the dispatch_once we are inside. */
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
    for (int i = 0; i < CONST_SLOTS; i++) { g_const[i].buf = nil; g_const[i].src = NULL; }
    g_const_count = 0;
    for (int i = 0; i < MM_SLOTS; i++) {
        g_mm[i].mm = nil; g_mm[i].dx = nil; g_mm[i].dW = nil; g_mm[i].dy = nil;
        g_mm[i].seq = g_mm[i].in_dim = g_mm[i].out_dim = g_mm[i].has_bias = 0;
    }
    g_mm_count = 0;
    g_attn = nil;
    g_queue = nil;
    g_device = nil;
    g_available = 0;
    pthread_mutex_unlock(&g_lock);
}

int granite_metal_available(void) { return g_available; }

int granite_metal_attn_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("GRANITE_METAL_ATTN");
        cached = (e && *e && *e != '0') ? 1 : 0;
    }
    return cached;
}

/* ========================================================================
 * Memory
 * ======================================================================== */

void *granite_metal_alloc(size_t bytes) {
    granite_metal_init();
    if (!g_available || bytes == 0) return NULL;

    pthread_mutex_lock(&g_lock);
    id<MTLBuffer> b = [g_device newBufferWithLength:bytes
                                           options:MTLResourceStorageModeShared];
    void *p = b ? [b contents] : NULL;
    /* If the range cannot be recorded, refuse: an unregistered pointer would
     * still work, but every GEMM touching it would stage a copy. */
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

/* ========================================================================
 * Dispatch policy
 * ======================================================================== */

int granite_metal_should_offload(int seq, int in_dim, int out_dim) {
    if (!g_available) return 0;
    if (seq <= 1) return 0;
    /* Below a few MFLOPs the dispatch round-trip dominates. Every linear in
     * this model clears this except on sub-second clips. */
    return (long long)seq * in_dim * out_dim * 2 >= 8000000LL;
}

/* ========================================================================
 * GEMM
 * ======================================================================== */

/* beta differs between the bias and no-bias cases, and it is baked into the
 * MPSMatrixMultiplication at construction, so it is part of the key. */
static int mm_slot(int seq, int in_dim, int out_dim, int has_bias) {
    for (int i = 0; i < g_mm_count; i++)
        if (g_mm[i].seq == seq && g_mm[i].in_dim == in_dim &&
            g_mm[i].out_dim == out_dim && g_mm[i].has_bias == has_bias)
            return i;
    if (g_mm_count == MM_SLOTS) g_mm_count = 0;   /* wrap; shapes recur in cycles */
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

/* Body of granite_metal_linear, minus the init. Kept separate so the warm-up
 * inside granite_metal_init() can reach it without re-entering dispatch_once,
 * which would deadlock. */
static int metal_linear(float *y, const float *x, const float *W,
                        const float *bias, int seq, int in_dim, int out_dim) {
    const size_t sx = (size_t)seq * in_dim * sizeof(float);
    const size_t sW = (size_t)out_dim * in_dim * sizeof(float);
    const size_t sy = (size_t)seq * out_dim * sizeof(float);

    size_t ox = 0, oW = 0, oy = 0;
    id<MTLBuffer> bx = operand(x, sx, 0, /*copy_in=*/1, &ox);
    id<MTLBuffer> bW = operand(W, sW, 1, /*copy_in=*/1, &oW);
    /* y is written, not read -- except that the bias prefill below seeds it,
     * which we do directly in whichever buffer it lands in. */
    id<MTLBuffer> by = operand(y, sy, 2, /*copy_in=*/0, &oy);
    if (!bx || !bW || !by) return 0;

    /* Bias goes in as beta=1 on a prefilled C, exactly like the BLAS path. */
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

/* ========================================================================
 * Block-local attention
 * ======================================================================== */

int granite_metal_block_attention(float *out, const float *q, const float *k,
                                  const float *v, int seq, int block_len,
                                  int heads, int dim_head, float scale,
                                  const float *rel, int n_pos,
                                  const int32_t *dists, int ctx) {
    granite_metal_init();
    if (!g_available) return 0;
    /* The kernel is written for one threadgroup per query with ATTN_WIDTH
     * threads covering both the key axis and the head dimension. */
    if (dim_head != MTL_ATTN_WIDTH || block_len > MTL_ATTN_WIDTH) return 0;
    if (block_len <= 0 || seq <= 0 || seq % block_len != 0) return 0;

    const int inner = heads * dim_head;
    const int n_blocks = seq / block_len;
    const size_t sqkv = (size_t)seq * inner * sizeof(float);
    const size_t srel = (size_t)n_pos * dim_head * sizeof(float);
    const size_t sdst = (size_t)ctx * ctx * sizeof(int32_t);

    size_t oq = 0, ok = 0, ov = 0, oo = 0, orel = 0;
    /* q/k/v/out are encoder scratch, so all four are resident and resolve to
     * offsets; the staging slots here are a correctness fallback only. */
    id<MTLBuffer> bq = reg_find(q, sqkv, &oq);
    id<MTLBuffer> bk = reg_find(k, sqkv, &ok);
    id<MTLBuffer> bv = reg_find(v, sqkv, &ov);
    id<MTLBuffer> bo = reg_find(out, sqkv, &oo);
    id<MTLBuffer> br = reg_find(rel, srel, &orel);
    if (!bq || !bk || !bv || !bo || !br) return 0;

    id<MTLBuffer> bd = const_upload(dists, sdst);
    if (!bd) return 0;

    attn_params_t P = {
        (uint32_t)block_len, (uint32_t)heads, (uint32_t)dim_head,
        (uint32_t)ctx, (uint32_t)inner, scale,
    };

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_attn];
    [enc setBuffer:bo offset:oo   atIndex:0];
    [enc setBuffer:bq offset:oq   atIndex:1];
    [enc setBuffer:bk offset:ok   atIndex:2];
    [enc setBuffer:bv offset:ov   atIndex:3];
    [enc setBuffer:br offset:orel atIndex:4];
    [enc setBuffer:bd offset:0    atIndex:5];
    [enc setBytes:&P length:sizeof(P) atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_blocks * heads * block_len, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MTL_ATTN_WIDTH, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return 1;
}
