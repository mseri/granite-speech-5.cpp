/* Math kernels for Granite Speech 5.0 inference. */

#ifndef GRANITE_KERNELS_H
#define GRANITE_KERNELS_H

#include <stddef.h>
#include <stdint.h>


void granite_set_threads(int n);
int granite_get_num_cpus(void);
void granite_kernels_shutdown(void);

/* Set the bf16-to-f32 weight-cache limit; zero disables caching. */
void granite_set_weight_cache_limit(size_t bytes);

/* Bytes currently held by the bf16 -> f32 weight cache. */
size_t granite_weight_cache_bytes(void);


/* Allocate a buffer for weights or activations. Under the Metal backend this
 * comes from GPU-visible shared memory, so kernels read the same bytes the CPU
 * wrote and no operand is ever uploaded; otherwise it is plain malloc.
 *
 * Buffers obtained here MUST be released with granite_device_free, not free().
 * granite_device_free(NULL) is a no-op. */
void *granite_device_alloc(size_t bytes);
void granite_device_free(void *p);


void granite_add_inplace(float *a, const float *b, int n);
void granite_scale(float *x, float s, int n);
float granite_bf16_to_f32(uint16_t h);

/* Compute a += s * b. */
void granite_add_scaled_inplace(float *a, const float *b, float s, int n);

/* Subsampling residual: out[t] = 0.5 * (x[2t] + x[2t+1]) + proj[t], for
 * t in [0, t_half). `out` must not alias `x`. */
void granite_subsample_pool_add(float *out, const float *x, const float *proj,
                                int t_half, int hidden);


/* y[seq, out] = x[seq, in] @ W[out, in]^T + b[out]. `b` may be NULL. */
void granite_linear_bf16(float *y, const float *x, const uint16_t *W,
                         const uint16_t *b, int seq, int in_dim, int out_dim);

/* Three bias-free linears sharing the same input x, e.g. Q/K/V projections.
 * Equivalent to three granite_linear_bf16 calls but, under the Metal
 * backend, dispatched as one command buffer instead of three round trips
 * (a bigger win with a discrete GPU's PCIe latency than a unified-memory one). */
void granite_linear3_bf16(float *y0, float *y1, float *y2, const float *x,
                          const uint16_t *W0, const uint16_t *W1,
                          const uint16_t *W2, int seq, int in_dim, int out_dim);


/* LayerNorm over the last dim, with bf16 affine params. `out` may alias `x`. */
void granite_layer_norm_bf16(float *out, const float *x, const uint16_t *w,
                             const uint16_t *b, int seq, int hidden, float eps);

/* Inference BatchNorm1d over channels, applied to x[seq, ch] (time-major).
 * out = (x - mean) / sqrt(var + eps) * w + b */
void granite_batch_norm_bf16(float *x, const uint16_t *w, const uint16_t *b,
                             const float *mean, const float *var,
                             int seq, int channels, float eps);


void granite_silu(float *x, int n);
void granite_softmax_rows(float *x, int rows, int cols);

/* GLU over the last dim: out[seq, half] = a * sigmoid(b) where the input row is
 * [a (half) | b (half)]. Operates out-of-place. */
void granite_glu(float *out, const float *x, int seq, int half);


/* Depthwise 1-D convolution, time-major in/out.
 * x:   [seq, ch]
 * w:   [ch, kernel]   (bf16, one filter per channel)
 * out: [out_seq, ch]  with out_seq = (seq + 2*pad - kernel)/stride + 1
 * Zero padding of `pad_l` on the left and `pad_r` on the right. */
void granite_depthwise_conv1d_bf16(float *out, const float *x, const uint16_t *w,
                                   int seq, int channels, int kernel,
                                   int stride, int pad_l, int pad_r);

/* --------------------------------------------------------------- attention -- */

/* Block-local multi-head attention with Shaw relative-position bias.
 *
 * Attention is computed independently over consecutive blocks of `block_len`
 * frames (no cross-block attention). Caller invokes this once per block group.
 *
 * q, k, v: [seq, heads * dim_head] for this group, seq = n_blocks * block_len
 * out:     [seq, heads * dim_head]
 * rel_pos_emb: [n_pos, dim_head] bf16 embedding table
 * dists:   [ctx * ctx] index table; the top-left [block_len, block_len]
 *          sub-window is used, matching the reference's slicing.
 * ctx:     stride of `dists` (GRANITE_CONTEXT_SIZE) */
void granite_block_attention(float *out, const float *q, const float *k,
                             const float *v, int seq, int block_len,
                             int heads, int dim_head, float scale,
                             const uint16_t *rel_pos_emb,
                             const int32_t *dists, int ctx);

#endif
