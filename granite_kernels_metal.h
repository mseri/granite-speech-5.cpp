/*
 * granite_kernels_metal.h - Metal/MPS backend (Apple Silicon)
 *
 * C-callable interface; safe to include from plain C translation units.
 * Implementation is in granite_kernels_metal.m (Objective-C + MPS + one
 * hand-written compute shader).
 *
 * Two things are offloaded, because the profile says they are the only two that
 * matter (see "Performance Notes" in CLAUDE.md):
 *   - every linear, via MPSMatrixMultiplication
 *   - block-local attention, via a custom kernel
 * Norms, GLU, SiLU and the depthwise conv stay on the CPU: they are a small
 * slice of the run and each would cost a dispatch round-trip.
 *
 * The design point that makes this worthwhile is that activations and the
 * bf16 -> f32 weight cache are allocated *as* Metal shared buffers, so the GPU
 * reads the same bytes the CPU wrote. There is no per-call upload of a 16 MB
 * weight matrix and no second copy of the 1.8 GB weight cache.
 */

#ifndef GRANITE_KERNELS_METAL_H
#define GRANITE_KERNELS_METAL_H

#include <stddef.h>
#include <stdint.h>

/* ---------------------------------------------------------------- lifecycle -- */

/*
 * Stderr verbosity, which exists to keep the user-facing output contract:
 * without --debug stderr is one summary line, and --silent quiets it entirely.
 *
 *   0  say nothing
 *   1  warnings only -- no device, shader compile failure (the default; these
 *      are worth a line because an MPS build silently running on the CPU is a
 *      real problem, and in the normal case nothing is printed)
 *   2  also report the device being used
 *
 * Must be set before the first call into the backend to have any effect on
 * init-time messages.
 */
void granite_metal_set_verbose(int level);

/* Bring up the device, queue and shader library. Idempotent and thread-safe;
 * safe to call before every other function here. */
void granite_metal_init(void);

/* Release every Metal resource, including registered buffers. Pointers handed
 * out by granite_metal_alloc() are dangling afterwards.
 *
 * Nothing in the CLI path calls this: device state is process-lifetime, exactly
 * like the bf16 -> f32 weight cache it backs, and both are reclaimed at exit.
 * It exists for embedders that load and unload a model inside a longer-lived
 * process. */
void granite_metal_shutdown(void);

/* 1 once init has succeeded, 0 if there is no usable Metal device. */
int granite_metal_available(void);

/* ------------------------------------------------------------------ memory -- */

/*
 * Allocate `bytes` of shared (CPU+GPU) storage and return a host pointer to it.
 * The backing MTLBuffer is registered, so any pointer into this range is
 * recognized later by the GEMM and attention entry points and used in place
 * without a staging copy.
 *
 * Returns NULL if Metal is unavailable or the allocation fails; callers are
 * expected to fall back to malloc.
 */
void *granite_metal_alloc(size_t bytes);

/* Free a pointer obtained from granite_metal_alloc(). Returns 1 if `p` was a
 * registered base pointer (and is now released), 0 if it was not ours -- in
 * which case the caller still owns it and should free() it. NULL returns 0. */
int granite_metal_dealloc(void *p);

/* ----------------------------------------------------------------- dispatch -- */

/* 1 when a seq x in_dim x out_dim linear is worth a GPU dispatch. Small
 * matrices lose to dispatch latency even on unified memory. */
int granite_metal_should_offload(int seq, int in_dim, int out_dim);

/* 1 when granite_metal_block_attention should be preferred over the CPU kernel.
 * Off unless GRANITE_METAL_ATTN is set to something other than 0: on an M1 the
 * shader measured 1.00 s against the CPU kernel's 0.27 s over a 119 s clip.
 * See the comment at the call site in granite_kernels.c. */
int granite_metal_attn_enabled(void);

/* -------------------------------------------------------------------- gemm -- */

/*
 * y[seq, out_dim] = x[seq, in_dim] @ W[out_dim, in_dim]^T + bias[out_dim]
 *
 * `W` and `bias` are already f32 (the caller converts bf16 once, into a
 * granite_metal_alloc'd buffer). `bias` may be NULL. All row-major.
 *
 * Returns 1 if the result was computed, 0 if the caller must fall back to CPU.
 */
int granite_metal_linear(float *y, const float *x, const float *W,
                         const float *bias, int seq, int in_dim, int out_dim);

/* --------------------------------------------------------------- attention -- */

/*
 * Block-local multi-head attention with Shaw relative-position bias -- the GPU
 * twin of granite_block_attention(). Same shapes and semantics, except `rel`
 * arrives already converted to f32 and its row count `n_pos` is explicit.
 *
 * Returns 1 if the result was computed, 0 if the caller must fall back to CPU
 * (unsupported shape, or no device).
 */
int granite_metal_block_attention(float *out, const float *q, const float *k,
                                  const float *v, int seq, int block_len,
                                  int heads, int dim_head, float scale,
                                  const float *rel, int n_pos,
                                  const int32_t *dists, int ctx);

#endif /* GRANITE_KERNELS_METAL_H */
