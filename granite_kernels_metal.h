/* C interface to the Metal/MPS backend. */

#ifndef GRANITE_KERNELS_METAL_H
#define GRANITE_KERNELS_METAL_H

#include <stddef.h>
#include <stdint.h>



/* Set backend diagnostics: 0 silent, 1 warnings, 2 includes the device name.
 * Set it before initialization. */
void granite_metal_set_verbose(int level);

/* Initialize the device and command queue. Safe to call repeatedly. */
void granite_metal_init(void);

/* Release Metal resources. Allocated pointers become invalid. */
void granite_metal_shutdown(void);

/* 1 once init has succeeded, 0 if there is no usable Metal device. */
int granite_metal_available(void);



/* Allocate shared CPU/GPU storage and return its host pointer, or NULL. */
void *granite_metal_alloc(size_t bytes);

/* Free a pointer obtained from granite_metal_alloc(). Returns 1 if `p` was a
 * registered base pointer (and is now released), 0 if it was not ours, in which
 * case the caller still owns it and should free() it. NULL returns 0. */
int granite_metal_dealloc(void *p);



/* 1 when a seq x in_dim x out_dim linear is worth a GPU dispatch. Small
 * matrices lose to dispatch latency; on a discrete GPU the PCIe copy of
 * staged operands raises that bar further. */
int granite_metal_should_offload(int seq, int in_dim, int out_dim);



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

#endif
