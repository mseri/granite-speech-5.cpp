/* Math kernels for Granite Speech 5.0 inference. */

#include "granite_kernels.h"
#include "granite.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef USE_MPS
#include "granite_kernels_metal.h"
#endif

/* Allocate shared CPU/GPU buffers when Metal is enabled. */

void *granite_device_alloc(size_t bytes) {
#ifdef USE_MPS
    void *p = granite_metal_alloc(bytes);
    if (p) return p;
#endif
    return malloc(bytes);
}

void granite_device_free(void *p) {
#ifdef USE_MPS
    if (granite_metal_dealloc(p)) return;
#endif
    free(p);
}


#define GRANITE_MAX_THREADS 32

typedef void (*parallel_fn_t)(int tid, int n_threads, void *arg);

static struct {
    pthread_t threads[GRANITE_MAX_THREADS - 1];
    int tids[GRANITE_MAX_THREADS - 1];
    int n_threads;
    int shutdown;

    parallel_fn_t fn;
    void *arg;
    int generation;

    pthread_mutex_t mutex;
    pthread_cond_t cond_work;
    pthread_cond_t cond_done;
    int n_done;
} tp = {
    .n_threads = 1,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .cond_work = PTHREAD_COND_INITIALIZER,
    .cond_done = PTHREAD_COND_INITIALIZER,
};

static void *worker_loop(void *arg) {
    int tid = *(int *)arg;
    int my_gen = 0;

    for (;;) {
        pthread_mutex_lock(&tp.mutex);
        while (tp.generation == my_gen && !tp.shutdown)
            pthread_cond_wait(&tp.cond_work, &tp.mutex);
        if (tp.shutdown) {
            pthread_mutex_unlock(&tp.mutex);
            return NULL;
        }
        my_gen = tp.generation;
        parallel_fn_t fn = tp.fn;
        void *a = tp.arg;
        int nt = tp.n_threads;
        pthread_mutex_unlock(&tp.mutex);

        fn(tid, nt, a);

        pthread_mutex_lock(&tp.mutex);
        if (++tp.n_done >= tp.n_threads - 1)
            pthread_cond_signal(&tp.cond_done);
        pthread_mutex_unlock(&tp.mutex);
    }
}

void granite_kernels_shutdown(void) {
    if (tp.n_threads <= 1) return;
    pthread_mutex_lock(&tp.mutex);
    tp.shutdown = 1;
    pthread_cond_broadcast(&tp.cond_work);
    pthread_mutex_unlock(&tp.mutex);
    for (int i = 0; i < tp.n_threads - 1; i++)
        pthread_join(tp.threads[i], NULL);
    tp.shutdown = 0;
    tp.generation = 0;
    tp.n_threads = 1;
}

void granite_set_threads(int n) {
    if (n < 1) n = 1;
    if (n > GRANITE_MAX_THREADS) n = GRANITE_MAX_THREADS;

    granite_kernels_shutdown();

    tp.n_threads = n;
    if (n <= 1) return;

    for (int i = 0; i < n - 1; i++) {
        tp.tids[i] = i + 1;
        pthread_create(&tp.threads[i], NULL, worker_loop, &tp.tids[i]);
    }
}

int granite_get_num_cpus(void) {
#ifdef __APPLE__
    int n = 0;
    size_t len = sizeof(n);
    sysctlbyname("hw.ncpu", &n, &len, NULL, 0);
    return n > 0 ? n : 1;
#else
    int n = (int)sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? n : 1;
#endif
}

static void parallel_for(parallel_fn_t fn, void *arg) {
    if (tp.n_threads <= 1) {
        fn(0, 1, arg);
        return;
    }

    pthread_mutex_lock(&tp.mutex);
    tp.fn = fn;
    tp.arg = arg;
    tp.n_done = 0;
    tp.generation++;
    pthread_cond_broadcast(&tp.cond_work);
    pthread_mutex_unlock(&tp.mutex);

    fn(0, tp.n_threads, arg);

    pthread_mutex_lock(&tp.mutex);
    while (tp.n_done < tp.n_threads - 1)
        pthread_cond_wait(&tp.cond_done, &tp.mutex);
    pthread_mutex_unlock(&tp.mutex);
}


float granite_bf16_to_f32(uint16_t h) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

/* Thread the encoder's elementwise residual adds. */
typedef struct {
    float *a;
    const float *b;
    float s;
    int n;
} axpy_args_t;

static void axpy_range(float *a, const float *b, float s, int start, int end) {
    int i = start;
#ifdef __ARM_NEON
    float32x4_t vs = vdupq_n_f32(s);
    for (; i + 16 <= end; i += 16) {
        vst1q_f32(a + i,      vfmaq_f32(vld1q_f32(a + i),
                                        vld1q_f32(b + i), vs));
        vst1q_f32(a + i + 4,  vfmaq_f32(vld1q_f32(a + i + 4),
                                        vld1q_f32(b + i + 4), vs));
        vst1q_f32(a + i + 8,  vfmaq_f32(vld1q_f32(a + i + 8),
                                        vld1q_f32(b + i + 8), vs));
        vst1q_f32(a + i + 12, vfmaq_f32(vld1q_f32(a + i + 12),
                                        vld1q_f32(b + i + 12), vs));
    }
#endif
    for (; i < end; i++) a[i] += s * b[i];
}

static void axpy_worker(int tid, int n_threads, void *arg) {
    axpy_args_t *a = arg;
    /* Multiple of 16 so every chunk keeps the unrolled NEON path. */
    int per = (((a->n + n_threads - 1) / n_threads) + 15) & ~15;
    int start = tid * per;
    if (start >= a->n) return;
    int end = start + per;
    if (end > a->n) end = a->n;
    axpy_range(a->a, a->b, a->s, start, end);
}

/* Dispatch costs more than the work below roughly a 32 Ki element buffer. */
#define AXPY_MIN_PARALLEL (1 << 15)

void granite_add_scaled_inplace(float *a, const float *b, float s, int n) {
    if (n < AXPY_MIN_PARALLEL) {
        axpy_range(a, b, s, 0, n);
        return;
    }
    axpy_args_t args = { a, b, s, n };
    parallel_for(axpy_worker, &args);
}

void granite_add_inplace(float *a, const float *b, int n) {
    granite_add_scaled_inplace(a, b, 1.0f, n);
}

void granite_scale(float *x, float s, int n) {
    for (int i = 0; i < n; i++) x[i] *= s;
}

/* Mean-pool a residual to half rate and add the conv output:
 *   out[t] = 0.5 * (x[2t] + x[2t+1]) + proj[t]
 *
 * Out-of-place on purpose. The serial version wrote back into x, which was safe
 * only because row t never overtakes source row 2t; once the t loop is split
 * across threads that ordering is gone and a thread writing row t can clobber
 * rows another thread has not read yet. `out` must not alias `x`. */
typedef struct {
    float *out;
    const float *x, *proj;
    int t_half, hidden;
} pool_args_t;

static void pool_worker(int tid, int n_threads, void *arg) {
    pool_args_t *a = arg;
    int per = (a->t_half + n_threads - 1) / n_threads;
    int start = tid * per;
    int end = start + per;
    if (end > a->t_half) end = a->t_half;

    for (int t = start; t < end; t++) {
        float *dst = a->out + (size_t)t * a->hidden;
        const float *s0 = a->x + (size_t)(2 * t) * a->hidden;
        const float *s1 = s0 + a->hidden;
        const float *c = a->proj + (size_t)t * a->hidden;
        int j = 0;
#ifdef __ARM_NEON
        /* 0.5 is a power of two so the scaling is exact either way; this keeps
         * the same single rounding of (s0 + s1) then of the add to c. */
        float32x4_t half = vdupq_n_f32(0.5f);
        for (; j + 4 <= a->hidden; j += 4)
            vst1q_f32(dst + j,
                      vfmaq_f32(vld1q_f32(c + j),
                                vaddq_f32(vld1q_f32(s0 + j), vld1q_f32(s1 + j)),
                                half));
#endif
        for (; j < a->hidden; j++)
            dst[j] = 0.5f * (s0[j] + s1[j]) + c[j];
    }
}

void granite_subsample_pool_add(float *out, const float *x, const float *proj,
                                int t_half, int hidden) {
    pool_args_t args = { out, x, proj, t_half, hidden };
    parallel_for(pool_worker, &args);
}

static void bf16_to_f32_range(float *dst, const uint16_t *src, size_t n) {
    size_t i = 0;
#ifdef __ARM_NEON
    for (; i + 8 <= n; i += 8) {
        uint16x8_t h = vld1q_u16(src + i);
        uint32x4_t lo = vshll_n_u16(vget_low_u16(h), 16);
        uint32x4_t hi = vshll_n_u16(vget_high_u16(h), 16);
        vst1q_f32(dst + i, vreinterpretq_f32_u32(lo));
        vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(hi));
    }
#endif
    for (; i < n; i++) dst[i] = granite_bf16_to_f32(src[i]);
}

/* Convert large weight tensors in parallel so conversion does not bottleneck
 * either backend. */
typedef struct {
    float *dst;
    const uint16_t *src;
    size_t n;
} conv_args_t;

static void conv_worker(int tid, int n_threads, void *arg) {
    conv_args_t *a = arg;
    /* Split on a multiple of 8 so each chunk keeps the NEON fast path. */
    size_t per = ((a->n + n_threads - 1) / n_threads + 7) & ~(size_t)7;
    size_t start = (size_t)tid * per;
    if (start >= a->n) return;
    size_t end = start + per;
    if (end > a->n) end = a->n;
    bf16_to_f32_range(a->dst + start, a->src + start, end - start);
}

static void bf16_to_f32_buf(float *dst, const uint16_t *src, size_t n) {
    /* Biases and LayerNorm affines are tiny; dispatching them costs more than
     * converting them. */
    if (n < (1u << 16)) {
        bf16_to_f32_range(dst, src, n);
        return;
    }
    conv_args_t a = { dst, src, n };
    parallel_for(conv_worker, &a);
}

/* Cache converted weights by their stable mapped source pointer. */

#define WCACHE_MAX_ENTRIES 1024

typedef struct {
    const uint16_t *src;
    float *f32;
    size_t n;
} wcache_entry_t;

static struct {
    wcache_entry_t entries[WCACHE_MAX_ENTRIES];
    int count;
    size_t used;
    size_t limit;
    int limit_set;
    pthread_mutex_t mutex;
} wcache = { .mutex = PTHREAD_MUTEX_INITIALIZER };

void granite_set_weight_cache_limit(size_t bytes) {
    pthread_mutex_lock(&wcache.mutex);
    wcache.limit = bytes;
    wcache.limit_set = 1;
    pthread_mutex_unlock(&wcache.mutex);
}

size_t granite_weight_cache_bytes(void) {
    pthread_mutex_lock(&wcache.mutex);
    size_t used = wcache.used;
    pthread_mutex_unlock(&wcache.mutex);
    return used;
}

/* Shared fallback buffer for weights that exceed the cache limit. */
static float *scratch = NULL;
static size_t scratch_n = 0;

static const float *bf16_as_f32(const uint16_t *src, size_t n) {
    pthread_mutex_lock(&wcache.mutex);
    if (!wcache.limit_set) {
        wcache.limit = (size_t)3 << 30;   /* 3 GiB: holds the whole model */
        wcache.limit_set = 1;
    }
    for (int i = 0; i < wcache.count; i++) {
        if (wcache.entries[i].src == src && wcache.entries[i].n == n) {
            const float *hit = wcache.entries[i].f32;
            pthread_mutex_unlock(&wcache.mutex);
            return hit;
        }
    }
    size_t bytes = n * sizeof(float);
    if (wcache.count < WCACHE_MAX_ENTRIES && wcache.used + bytes <= wcache.limit) {
        float *buf = granite_device_alloc(bytes);
        if (buf) {
            bf16_to_f32_buf(buf, src, n);
            wcache.entries[wcache.count].src = src;
            wcache.entries[wcache.count].f32 = buf;
            wcache.entries[wcache.count].n = n;
            wcache.count++;
            wcache.used += bytes;
            pthread_mutex_unlock(&wcache.mutex);
            return buf;
        }
    }
    /* Use shared scratch storage for uncached weights. */
    if (scratch_n < n) {
        granite_device_free(scratch);
        scratch = granite_device_alloc(bytes);
        scratch_n = scratch ? n : 0;
    }
    const float *out = scratch;
    if (scratch) bf16_to_f32_buf(scratch, src, n);
    pthread_mutex_unlock(&wcache.mutex);
    return out;
}


/* Add one bias row to every frame of a GEMM result. */
typedef struct {
    float *y;
    const float *bias;
    int seq, out_dim;
} bias_args_t;

static void bias_worker(int tid, int n_threads, void *arg) {
    bias_args_t *a = arg;
    int per = (a->seq + n_threads - 1) / n_threads;
    int start = tid * per;
    int end = start + per;
    if (end > a->seq) end = a->seq;

    for (int i = start; i < end; i++) {
        float *row = a->y + (size_t)i * a->out_dim;
        int o = 0;
#ifdef __ARM_NEON
        for (; o + 16 <= a->out_dim; o += 16) {
            vst1q_f32(row + o,      vaddq_f32(vld1q_f32(row + o),
                                              vld1q_f32(a->bias + o)));
            vst1q_f32(row + o + 4,  vaddq_f32(vld1q_f32(row + o + 4),
                                              vld1q_f32(a->bias + o + 4)));
            vst1q_f32(row + o + 8,  vaddq_f32(vld1q_f32(row + o + 8),
                                              vld1q_f32(a->bias + o + 8)));
            vst1q_f32(row + o + 12, vaddq_f32(vld1q_f32(row + o + 12),
                                              vld1q_f32(a->bias + o + 12)));
        }
#endif
        for (; o < a->out_dim; o++) row[o] += a->bias[o];
    }
}

static void add_bias_rows(float *y, const float *bias, int seq, int out_dim) {
    bias_args_t a = { y, bias, seq, out_dim };
    parallel_for(bias_worker, &a);
}

#ifndef USE_BLAS
typedef struct {
    float *y;
    const float *x, *W, *b;
    int seq, in_dim, out_dim;
} linear_args_t;

static void linear_worker(int tid, int n_threads, void *arg) {
    linear_args_t *a = arg;
    int rows_per = (a->seq + n_threads - 1) / n_threads;
    int start = tid * rows_per;
    int end = start + rows_per;
    if (end > a->seq) end = a->seq;

    for (int i = start; i < end; i++) {
        const float *xi = a->x + (size_t)i * a->in_dim;
        float *yi = a->y + (size_t)i * a->out_dim;
        for (int o = 0; o < a->out_dim; o++) {
            const float *w = a->W + (size_t)o * a->in_dim;
            float acc = a->b ? a->b[o] : 0.0f;
#ifdef __ARM_NEON
            float32x4_t v = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 4 <= a->in_dim; k += 4)
                v = vfmaq_f32(v, vld1q_f32(xi + k), vld1q_f32(w + k));
            acc += vaddvq_f32(v);
            for (; k < a->in_dim; k++) acc += xi[k] * w[k];
#else
            for (int k = 0; k < a->in_dim; k++) acc += xi[k] * w[k];
#endif
            yi[o] = acc;
        }
    }
}
#endif

void granite_linear_bf16(float *y, const float *x, const uint16_t *W,
                         const uint16_t *b, int seq, int in_dim, int out_dim) {
    const float *Wf = bf16_as_f32(W, (size_t)in_dim * out_dim);
    if (!Wf) return;

    float bias_stack[GRANITE_HIDDEN * GRANITE_FF_MULT];
    float *bias = NULL;
    if (b) {
        /* Bias vectors are at most 4*HIDDEN except for the CTC head. */
        bias = (out_dim <= GRANITE_HIDDEN * GRANITE_FF_MULT)
                   ? bias_stack : malloc((size_t)out_dim * sizeof(float));
        if (!bias) return;
        for (int o = 0; o < out_dim; o++) bias[o] = granite_bf16_to_f32(b[o]);
    }

#ifdef USE_MPS
    /* Fall back to CPU for small shapes or unregistered operands. Metal keeps
     * the bias prefill because it also initializes GPU-written pages. */
    if (granite_metal_should_offload(seq, in_dim, out_dim) &&
        granite_metal_linear(y, x, Wf, bias, seq, in_dim, out_dim))
        goto done;
#endif

#ifdef USE_BLAS
    /* beta=0 avoids reading C; add the bias in a separate parallel pass. */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq, out_dim, in_dim, 1.0f,
                x, in_dim, Wf, in_dim,
                0.0f, y, out_dim);
    if (bias) add_bias_rows(y, bias, seq, out_dim);
#else
    linear_args_t args = { y, x, Wf, bias, seq, in_dim, out_dim };
    parallel_for(linear_worker, &args);
#endif

#ifdef USE_MPS
done:
#endif
    if (bias && bias != bias_stack) free(bias);
}


typedef struct {
    float *out;
    const float *x;
    const float *w, *b;
    int seq, hidden;
    float eps;
} ln_args_t;

static void layer_norm_worker(int tid, int n_threads, void *arg) {
    ln_args_t *a = arg;
    int rows_per = (a->seq + n_threads - 1) / n_threads;
    int start = tid * rows_per;
    int end = start + rows_per;
    if (end > a->seq) end = a->seq;

    for (int i = start; i < end; i++) {
        const float *xi = a->x + (size_t)i * a->hidden;
        float *oi = a->out + (size_t)i * a->hidden;
        float mean = 0.0f;
        for (int j = 0; j < a->hidden; j++) mean += xi[j];
        mean /= a->hidden;
        float var = 0.0f;
        for (int j = 0; j < a->hidden; j++) {
            float d = xi[j] - mean;
            var += d * d;
        }
        var /= a->hidden;
        float inv = 1.0f / sqrtf(var + a->eps);
        for (int j = 0; j < a->hidden; j++)
            oi[j] = (xi[j] - mean) * inv * a->w[j] + a->b[j];
    }
}

void granite_layer_norm_bf16(float *out, const float *x, const uint16_t *w,
                             const uint16_t *b, int seq, int hidden, float eps) {
    if (hidden > GRANITE_HIDDEN) return;
    /* Two live conversions at once would both land in the shared scratch, so
     * copy the first out before requesting the second. */
    float wbuf[GRANITE_HIDDEN];
    float bbuf[GRANITE_HIDDEN];
    const float *wf = bf16_as_f32(w, hidden);
    if (!wf) return;
    memcpy(wbuf, wf, (size_t)hidden * sizeof(float));
    const float *bf = bf16_as_f32(b, hidden);
    if (!bf) return;
    memcpy(bbuf, bf, (size_t)hidden * sizeof(float));

    ln_args_t args = { out, x, wbuf, bbuf, seq, hidden, eps };
    parallel_for(layer_norm_worker, &args);
}

typedef struct {
    float *x;
    const float *scale, *shift;
    int seq, channels;
} bn_args_t;

static void bn_worker(int tid, int n_threads, void *arg) {
    bn_args_t *a = arg;
    int per = (a->seq + n_threads - 1) / n_threads;
    int start = tid * per;
    int end = start + per;
    if (end > a->seq) end = a->seq;

    for (int t = start; t < end; t++) {
        float *xt = a->x + (size_t)t * a->channels;
        for (int c = 0; c < a->channels; c++)
            xt[c] = xt[c] * a->scale[c] + a->shift[c];
    }
}

void granite_batch_norm_bf16(float *x, const uint16_t *w, const uint16_t *b,
                             const float *mean, const float *var,
                             int seq, int channels, float eps) {
    /* Fold the affine and the statistics into a single scale/shift per channel
     * so the inner loop over frames is one FMA. */
    float *scale = malloc((size_t)channels * 2 * sizeof(float));
    if (!scale) return;
    float *shift = scale + channels;
    for (int c = 0; c < channels; c++) {
        float g = granite_bf16_to_f32(w[c]);
        float inv = 1.0f / sqrtf(var[c] + eps);
        scale[c] = g * inv;
        shift[c] = granite_bf16_to_f32(b[c]) - mean[c] * scale[c];
    }
    bn_args_t args = { x, scale, shift, seq, channels };
    parallel_for(bn_worker, &args);
    free(scale);
}


/* Thread activation work and use vForce exp where available. */
#define EW_TILE 512

static void exp_tile(float *dst, const float *src, int n) {
#if defined(USE_BLAS) && defined(__APPLE__)
    int nn = n;
    vvexpf(dst, src, &nn);
#else
    for (int i = 0; i < n; i++) dst[i] = expf(src[i]);
#endif
}

typedef struct {
    float *x;
    int n;
} silu_args_t;

static void silu_worker(int tid, int n_threads, void *arg) {
    silu_args_t *a = arg;
    int per = (a->n + n_threads - 1) / n_threads;
    int start = tid * per;
    int end = start + per;
    if (end > a->n) end = a->n;

    float neg[EW_TILE], ex[EW_TILE];
    for (int i = start; i < end; i += EW_TILE) {
        int m = (end - i < EW_TILE) ? end - i : EW_TILE;
        for (int j = 0; j < m; j++) neg[j] = -a->x[i + j];
        exp_tile(ex, neg, m);
        for (int j = 0; j < m; j++) a->x[i + j] /= 1.0f + ex[j];
    }
}

void granite_silu(float *x, int n) {
    silu_args_t a = { x, n };
    parallel_for(silu_worker, &a);
}

typedef struct {
    float *x;
    int rows, cols;
} softmax_args_t;

static void softmax_worker(int tid, int n_threads, void *arg) {
    softmax_args_t *a = arg;
    int per = (a->rows + n_threads - 1) / n_threads;
    int start = tid * per;
    int end = start + per;
    if (end > a->rows) end = a->rows;

    float shifted[EW_TILE];
    for (int r = start; r < end; r++) {
        float *row = a->x + (size_t)r * a->cols;
        float mx = row[0];
        for (int c = 1; c < a->cols; c++) if (row[c] > mx) mx = row[c];
        for (int c = 0; c < a->cols; c += EW_TILE) {
            int m = (a->cols - c < EW_TILE) ? a->cols - c : EW_TILE;
            for (int j = 0; j < m; j++) shifted[j] = row[c + j] - mx;
            exp_tile(row + c, shifted, m);
        }
        /* Preserve the reference's accumulation order. */
        float sum = 0.0f;
        for (int c = 0; c < a->cols; c++) sum += row[c];
        float inv = 1.0f / sum;
        for (int c = 0; c < a->cols; c++) row[c] *= inv;
    }
}

void granite_softmax_rows(float *x, int rows, int cols) {
    softmax_args_t a = { x, rows, cols };
    parallel_for(softmax_worker, &a);
}

typedef struct {
    float *out;
    const float *x;
    int seq, half;
} glu_args_t;

static void glu_worker(int tid, int n_threads, void *arg) {
    glu_args_t *a = arg;
    int per = (a->seq + n_threads - 1) / n_threads;
    int start = tid * per;
    int end = start + per;
    if (end > a->seq) end = a->seq;

    float neg[EW_TILE], ex[EW_TILE];
    for (int t = start; t < end; t++) {
        const float *av = a->x + (size_t)t * a->half * 2;
        const float *g = av + a->half;
        float *o = a->out + (size_t)t * a->half;
        for (int i = 0; i < a->half; i += EW_TILE) {
            int m = (a->half - i < EW_TILE) ? a->half - i : EW_TILE;
            for (int j = 0; j < m; j++) neg[j] = -g[i + j];
            exp_tile(ex, neg, m);
            for (int j = 0; j < m; j++) o[i + j] = av[i + j] / (1.0f + ex[j]);
        }
    }
}

void granite_glu(float *out, const float *x, int seq, int half) {
    glu_args_t a = { out, x, seq, half };
    parallel_for(glu_worker, &a);
}


typedef struct {
    float *out;
    const float *x, *wf;
    int seq, channels, kernel, stride, pad_l, out_seq;
} dw_args_t;

static void dw_worker(int tid, int n_threads, void *arg) {
    dw_args_t *a = arg;
    int per = (a->out_seq + n_threads - 1) / n_threads;
    int start = tid * per;
    int end = start + per;
    if (end > a->out_seq) end = a->out_seq;

    for (int o = start; o < end; o++) {
        float *dst = a->out + (size_t)o * a->channels;
        memset(dst, 0, (size_t)a->channels * sizeof(float));
        int base = o * a->stride - a->pad_l;
        for (int j = 0; j < a->kernel; j++) {
            int t = base + j;
            if (t < 0 || t >= a->seq) continue;      /* zero padding */
            const float *src = a->x + (size_t)t * a->channels;
            for (int c = 0; c < a->channels; c++)
                dst[c] += src[c] * a->wf[(size_t)c * a->kernel + j];
        }
    }
}

void granite_depthwise_conv1d_bf16(float *out, const float *x, const uint16_t *w,
                                   int seq, int channels, int kernel,
                                   int stride, int pad_l, int pad_r) {
    int out_seq = (seq + pad_l + pad_r - kernel) / stride + 1;
    const float *wf = bf16_as_f32(w, (size_t)channels * kernel);
    if (!wf) return;

    /* Output rows are independent, so the convolution is safe to thread. */
    dw_args_t args = { out, x, wf, seq, channels, kernel, stride, pad_l, out_seq };
    parallel_for(dw_worker, &args);
}


typedef struct {
    float *out;
    const float *q, *k, *v;
    int n_blocks, block_len, heads, dim_head;
    float scale;
    const float *rel;          /* [n_pos, dim_head] as f32 */
    const int32_t *dists;
    int ctx;
} attn_args_t;

/*
 * Query blocking (QBLK). The flops here are trivial; the latency is the
 * problem. One query row against one key is a 128-element dot reduced into a
 * single NEON accumulator, a 32-long chain of 4-cycle FMAs that uses one of the
 * four FMA pipes. Measured, the whole kernel ran at ~53 GFLOP/s while
 * Accelerate's sgemm reaches ~1000 on the same machine.
 *
 * The fix is to run QBLK query rows against each key at once. That yields
 * 2 * QBLK independent accumulators (content and Shaw position per query), and
 * each k_j / v_j load is shared across QBLK queries instead of being re-read.
 *
 * Every accumulator still sums over d in the same order as the scalar path did,
 * and the j loop still runs ascending, so this is bit-identical to the original,
 * which matters because the suite demands exact argmax agreement.
 */
#define QBLK 4

/* Normalize one score row and apply it to the values. */
static void attn_row_finish(float *row, int L, float *oi, const float *v,
                            int off, int inner, int hoff, int D) {
    float mx = -INFINITY;
    for (int j = 0; j < L; j++) if (row[j] > mx) mx = row[j];
    float sum = 0.0f;
    for (int j = 0; j < L; j++) {
        row[j] = expf(row[j] - mx);
        sum += row[j];
    }
    float inv = 1.0f / sum;

    memset(oi, 0, (size_t)D * sizeof(float));
    for (int j = 0; j < L; j++) {
        float wgt = row[j] * inv;
        const float *vj = v + (size_t)(off + j) * inner + hoff;
#ifdef __ARM_NEON
        float32x4_t vw = vdupq_n_f32(wgt);
        int d = 0;
        for (; d + 4 <= D; d += 4)
            vst1q_f32(oi + d, vfmaq_f32(vld1q_f32(oi + d), vld1q_f32(vj + d), vw));
        for (; d < D; d++) oi[d] += wgt * vj[d];
#else
        for (int d = 0; d < D; d++) oi[d] += wgt * vj[d];
#endif
    }
}

/* Score one query row for a short tail. */
static void attn_row_scores(const attn_args_t *a, float *row, int i,
                            int off, int inner, int hoff) {
    const int L = a->block_len, D = a->dim_head;
    const float *qi = a->q + (size_t)(off + i) * inner + hoff;
    for (int j = 0; j < L; j++) {
        const float *kj = a->k + (size_t)(off + j) * inner + hoff;
        const float *rel =
            a->rel + (size_t)a->dists[(size_t)i * a->ctx + j] * D;
        float dot = 0.0f, pos = 0.0f;
#ifdef __ARM_NEON
        float32x4_t vd = vdupq_n_f32(0.0f), vp = vdupq_n_f32(0.0f);
        int d = 0;
        for (; d + 4 <= D; d += 4) {
            float32x4_t vq = vld1q_f32(qi + d);
            vd = vfmaq_f32(vd, vq, vld1q_f32(kj + d));
            vp = vfmaq_f32(vp, vq, vld1q_f32(rel + d));
        }
        dot = vaddvq_f32(vd);
        pos = vaddvq_f32(vp);
        for (; d < D; d++) { dot += qi[d] * kj[d]; pos += qi[d] * rel[d]; }
#else
        for (int d = 0; d < D; d++) { dot += qi[d] * kj[d]; pos += qi[d] * rel[d]; }
#endif
        row[j] = (dot + pos) * a->scale;
    }
}

/* Score QBLK query rows together; keep QBLK literal for register allocation. */
static void attn_scores_x4(const attn_args_t *a, float *scores, int i0,
                           int off, int inner, int hoff) {
    const int L = a->block_len, D = a->dim_head;
    const float *qp[QBLK];
    for (int u = 0; u < QBLK; u++)
        qp[u] = a->q + (size_t)(off + i0 + u) * inner + hoff;

    for (int j = 0; j < L; j++) {
        const float *kj = a->k + (size_t)(off + j) * inner + hoff;
        const float *rp[QBLK];
        for (int u = 0; u < QBLK; u++)
            rp[u] = a->rel +
                (size_t)a->dists[(size_t)(i0 + u) * a->ctx + j] * D;
#ifdef __ARM_NEON
        float32x4_t vd0 = vdupq_n_f32(0.0f), vd1 = vdupq_n_f32(0.0f);
        float32x4_t vd2 = vdupq_n_f32(0.0f), vd3 = vdupq_n_f32(0.0f);
        float32x4_t vp0 = vdupq_n_f32(0.0f), vp1 = vdupq_n_f32(0.0f);
        float32x4_t vp2 = vdupq_n_f32(0.0f), vp3 = vdupq_n_f32(0.0f);
        int d = 0;
        for (; d + 4 <= D; d += 4) {
            float32x4_t vk = vld1q_f32(kj + d);
            float32x4_t q0 = vld1q_f32(qp[0] + d), q1 = vld1q_f32(qp[1] + d);
            float32x4_t q2 = vld1q_f32(qp[2] + d), q3 = vld1q_f32(qp[3] + d);
            vd0 = vfmaq_f32(vd0, q0, vk);
            vd1 = vfmaq_f32(vd1, q1, vk);
            vd2 = vfmaq_f32(vd2, q2, vk);
            vd3 = vfmaq_f32(vd3, q3, vk);
            vp0 = vfmaq_f32(vp0, q0, vld1q_f32(rp[0] + d));
            vp1 = vfmaq_f32(vp1, q1, vld1q_f32(rp[1] + d));
            vp2 = vfmaq_f32(vp2, q2, vld1q_f32(rp[2] + d));
            vp3 = vfmaq_f32(vp3, q3, vld1q_f32(rp[3] + d));
        }
        float dot[QBLK] = { vaddvq_f32(vd0), vaddvq_f32(vd1),
                            vaddvq_f32(vd2), vaddvq_f32(vd3) };
        float pos[QBLK] = { vaddvq_f32(vp0), vaddvq_f32(vp1),
                            vaddvq_f32(vp2), vaddvq_f32(vp3) };
        for (int u = 0; u < QBLK; u++) {
            for (int t = d; t < D; t++) {
                dot[u] += qp[u][t] * kj[t];
                pos[u] += qp[u][t] * rp[u][t];
            }
            scores[u * L + j] = (dot[u] + pos[u]) * a->scale;
        }
#else
        for (int u = 0; u < QBLK; u++) {
            float dot = 0.0f, pos = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += qp[u][d] * kj[d];
                pos += qp[u][d] * rp[u][d];
            }
            scores[u * L + j] = (dot + pos) * a->scale;
        }
#endif
    }
}

static void block_attn_worker(int tid, int n_threads, void *arg) {
    attn_args_t *a = arg;
    const int inner = a->heads * a->dim_head;
    const int total = a->n_blocks * a->heads;
    const int L = a->block_len, D = a->dim_head;
    float *scores = malloc((size_t)QBLK * L * sizeof(float));
    if (!scores) return;

    for (int job = tid; job < total; job += n_threads) {
        int blk = job / a->heads;
        int h = job % a->heads;
        int off = blk * L;                     /* first frame of this block */
        int hoff = h * D;

        int i = 0;
        for (; i + QBLK <= L; i += QBLK) {
            attn_scores_x4(a, scores, i, off, inner, hoff);
            for (int u = 0; u < QBLK; u++)
                attn_row_finish(scores + u * L, L,
                                a->out + (size_t)(off + i + u) * inner + hoff,
                                a->v, off, inner, hoff, D);
        }
        for (; i < L; i++) {
            attn_row_scores(a, scores, i, off, inner, hoff);
            attn_row_finish(scores, L,
                            a->out + (size_t)(off + i) * inner + hoff,
                            a->v, off, inner, hoff, D);
        }
    }
    free(scores);
}

void granite_block_attention(float *out, const float *q, const float *k,
                             const float *v, int seq, int block_len,
                             int heads, int dim_head, float scale,
                             const uint16_t *rel_pos_emb,
                             const int32_t *dists, int ctx) {
    if (block_len <= 0 || seq <= 0) return;
    int n_pos = 2 * GRANITE_MAX_POS_EMB + 1;
    const float *rel = bf16_as_f32(rel_pos_emb, (size_t)n_pos * dim_head);
    if (!rel) return;

    attn_args_t args = {
        out, q, k, v, seq / block_len, block_len, heads, dim_head,
        scale, rel, dists, ctx,
    };
    parallel_for(block_attn_worker, &args);
}
