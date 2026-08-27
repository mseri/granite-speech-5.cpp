/*
 * granite_kernels.c - Math kernels for Granite Speech 5.0 inference
 *
 * The thread pool is adapted from the qwen-asr project. Unlike qwen-asr's
 * decoder, every matmul here is a real GEMM over the frame axis (there is no
 * seq=1 step), so the linear path is bf16 -> f32 weight conversion feeding
 * cblas_sgemm, with converted weights cached across calls.
 */

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

/* ========================================================================
 * Allocation
 *
 * Under Metal these are shared (CPU+GPU) buffers, which is what keeps the
 * 1.8 GB weight cache from being duplicated on the device: the GPU reads the
 * exact bytes the bf16 -> f32 conversion wrote.
 * ======================================================================== */

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

/* ========================================================================
 * Thread pool (adapted from qwen-asr)
 * ======================================================================== */

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

/* ========================================================================
 * Basics
 * ======================================================================== */

float granite_bf16_to_f32(uint16_t h) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

void granite_add_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

void granite_scale(float *x, float s, int n) {
    for (int i = 0; i < n; i++) x[i] *= s;
}

static void bf16_to_f32_buf(float *dst, const uint16_t *src, size_t n) {
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

/* ========================================================================
 * bf16 -> f32 weight cache
 *
 * Every weight tensor is converted at most once and reused for the rest of the
 * run. Entries are keyed by source pointer, which is stable because the
 * checkpoint stays mmap'd for the model's lifetime.
 * ======================================================================== */

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

/* Scratch buffer used when a weight does not fit in the cache. Per-thread is
 * unnecessary: conversion happens on the calling thread before parallel_for. */
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
    /* Over the cap (or out of memory): convert into the shared scratch buffer.
     * Valid until the next uncached conversion, which is enough because callers
     * consume the pointer within a single kernel invocation. */
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

/* ========================================================================
 * Linear
 * ======================================================================== */

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
#endif /* !USE_BLAS */

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
    /* Falls through to the CPU path if the shape is too small to be worth a
     * dispatch, or if an operand could not be resolved to a device buffer. */
    if (granite_metal_should_offload(seq, in_dim, out_dim) &&
        granite_metal_linear(y, x, Wf, bias, seq, in_dim, out_dim))
        goto done;
#endif

#ifdef USE_BLAS
    if (bias) {
        for (int i = 0; i < seq; i++)
            memcpy(y + (size_t)i * out_dim, bias, (size_t)out_dim * sizeof(float));
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq, out_dim, in_dim, 1.0f,
                x, in_dim, Wf, in_dim,
                bias ? 1.0f : 0.0f, y, out_dim);
#else
    linear_args_t args = { y, x, Wf, bias, seq, in_dim, out_dim };
    parallel_for(linear_worker, &args);
#endif

#ifdef USE_MPS
done:
#endif
    if (bias && bias != bias_stack) free(bias);
}

/* ========================================================================
 * Normalization
 * ======================================================================== */

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
    for (int t = 0; t < seq; t++) {
        float *xt = x + (size_t)t * channels;
        for (int c = 0; c < channels; c++) xt[c] = xt[c] * scale[c] + shift[c];
    }
    free(scale);
}

/* ========================================================================
 * Activations
 * ======================================================================== */

void granite_silu(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

void granite_softmax_rows(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + (size_t)r * cols;
        float mx = row[0];
        for (int c = 1; c < cols; c++) if (row[c] > mx) mx = row[c];
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - mx);
            sum += row[c];
        }
        float inv = 1.0f / sum;
        for (int c = 0; c < cols; c++) row[c] *= inv;
    }
}

void granite_glu(float *out, const float *x, int seq, int half) {
    for (int t = 0; t < seq; t++) {
        const float *a = x + (size_t)t * half * 2;
        const float *g = a + half;
        float *o = out + (size_t)t * half;
        for (int i = 0; i < half; i++)
            o[i] = a[i] / (1.0f + expf(-g[i]));
    }
}

/* ========================================================================
 * Depthwise conv1d
 * ======================================================================== */

void granite_depthwise_conv1d_bf16(float *out, const float *x, const uint16_t *w,
                                   int seq, int channels, int kernel,
                                   int stride, int pad_l, int pad_r) {
    int out_seq = (seq + pad_l + pad_r - kernel) / stride + 1;
    const float *wf = bf16_as_f32(w, (size_t)channels * kernel);
    if (!wf) return;

    for (int o = 0; o < out_seq; o++) {
        float *dst = out + (size_t)o * channels;
        memset(dst, 0, (size_t)channels * sizeof(float));
        int base = o * stride - pad_l;
        for (int j = 0; j < kernel; j++) {
            int t = base + j;
            if (t < 0 || t >= seq) continue;      /* zero padding */
            const float *src = x + (size_t)t * channels;
            for (int c = 0; c < channels; c++)
                dst[c] += src[c] * wf[(size_t)c * kernel + j];
        }
    }
}

/* ========================================================================
 * Block-local attention with Shaw relative-position bias
 * ======================================================================== */

typedef struct {
    float *out;
    const float *q, *k, *v;
    int n_blocks, block_len, heads, dim_head;
    float scale;
    const float *rel;          /* [n_pos, dim_head] as f32 */
    const int32_t *dists;
    int ctx;
} attn_args_t;

static void block_attn_worker(int tid, int n_threads, void *arg) {
    attn_args_t *a = arg;
    int inner = a->heads * a->dim_head;
    int total = a->n_blocks * a->heads;
    float *scores = malloc((size_t)a->block_len * sizeof(float));
    if (!scores) return;

    for (int job = tid; job < total; job += n_threads) {
        int blk = job / a->heads;
        int h = job % a->heads;
        int off = blk * a->block_len;          /* first frame of this block */
        int hoff = h * a->dim_head;

        for (int i = 0; i < a->block_len; i++) {
            const float *qi = a->q + (size_t)(off + i) * inner + hoff;

            /* Content score + Shaw positional bias, both scaled. */
            float mx = -INFINITY;
            for (int j = 0; j < a->block_len; j++) {
                const float *kj = a->k + (size_t)(off + j) * inner + hoff;
                const float *rel = a->rel +
                    (size_t)a->dists[(size_t)i * a->ctx + j] * a->dim_head;
                float dot = 0.0f, pos = 0.0f;
#ifdef __ARM_NEON
                float32x4_t vd = vdupq_n_f32(0.0f), vp = vdupq_n_f32(0.0f);
                int d = 0;
                for (; d + 4 <= a->dim_head; d += 4) {
                    float32x4_t vq = vld1q_f32(qi + d);
                    vd = vfmaq_f32(vd, vq, vld1q_f32(kj + d));
                    vp = vfmaq_f32(vp, vq, vld1q_f32(rel + d));
                }
                dot = vaddvq_f32(vd);
                pos = vaddvq_f32(vp);
                for (; d < a->dim_head; d++) {
                    dot += qi[d] * kj[d];
                    pos += qi[d] * rel[d];
                }
#else
                for (int d = 0; d < a->dim_head; d++) {
                    dot += qi[d] * kj[d];
                    pos += qi[d] * rel[d];
                }
#endif
                float s = (dot + pos) * a->scale;
                scores[j] = s;
                if (s > mx) mx = s;
            }

            float sum = 0.0f;
            for (int j = 0; j < a->block_len; j++) {
                scores[j] = expf(scores[j] - mx);
                sum += scores[j];
            }
            float inv = 1.0f / sum;

            float *oi = a->out + (size_t)(off + i) * inner + hoff;
            memset(oi, 0, (size_t)a->dim_head * sizeof(float));
            for (int j = 0; j < a->block_len; j++) {
                float wgt = scores[j] * inv;
                const float *vj = a->v + (size_t)(off + j) * inner + hoff;
#ifdef __ARM_NEON
                float32x4_t vw = vdupq_n_f32(wgt);
                int d = 0;
                for (; d + 4 <= a->dim_head; d += 4)
                    vst1q_f32(oi + d, vfmaq_f32(vld1q_f32(oi + d),
                                                vld1q_f32(vj + d), vw));
                for (; d < a->dim_head; d++) oi[d] += wgt * vj[d];
#else
                for (int d = 0; d < a->dim_head; d++) oi[d] += wgt * vj[d];
#endif
            }
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

#ifdef USE_MPS
    if (granite_metal_block_attention(out, q, k, v, seq, block_len, heads,
                                      dim_head, scale, rel, n_pos, dists, ctx))
        return;
#endif

    attn_args_t args = {
        out, q, k, v, seq / block_len, block_len, heads, dim_head,
        scale, rel, dists, ctx,
    };
    parallel_for(block_attn_worker, &args);
}
