/*
 * granite.h - Public API for Granite Speech 5.0 CTC inference
 *
 * Encoder-only CTC model: audio -> log-mel(+delta) front-end -> conformer
 * encoder -> CTC head -> greedy collapse -> text. There is no autoregressive
 * decoder, no KV cache and no sampling.
 */

#ifndef GRANITE_H
#define GRANITE_H

#include <stddef.h>
#include <stdint.h>

#include "granite_safetensors.h"

/* ========================================================================
 * Model hyper-parameters (granite-speech-5.0/config.json)
 * ======================================================================== */

#define GRANITE_INPUT_DIM      320   /* n_mels * 2 (deltas) * stack_factor */
#define GRANITE_NUM_LAYERS      16
#define GRANITE_HIDDEN        1024
#define GRANITE_FF_MULT          4   /* feed-forward inner = HIDDEN * FF_MULT */
#define GRANITE_NUM_HEADS        8
#define GRANITE_DIM_HEAD       128
#define GRANITE_VOCAB        16384
#define GRANITE_CONTEXT_SIZE   128   /* attention block length */
#define GRANITE_MAX_POS_EMB    512   /* rel_pos_emb has 2*this+1 rows */
#define GRANITE_CONV_KERNEL      7
#define GRANITE_CONV_EXPANSION   2   /* conv inner = HIDDEN * CONV_EXPANSION */

/* Blocks 0 and 1 subsample time by 2 each (4x total). */
#define GRANITE_SUBSAMPLE_LAYERS 2

/* CTC blank shares id 0 with the tokenizer's <unk>. */
#define GRANITE_BLANK_ID 0

/* ========================================================================
 * Front-end parameters (preprocessor_config.json)
 * ======================================================================== */

#define GRANITE_SAMPLE_RATE    16000
#define GRANITE_N_FFT            512
#define GRANITE_WIN_LENGTH       400
#define GRANITE_HOP_LENGTH       160
#define GRANITE_N_MELS            80
#define GRANITE_STACK_FACTOR       2
#define GRANITE_DELTA_WIN_LENGTH   3
#define GRANITE_LOGMEL_FLOOR_DB  8.0f

/* One encoder frame spans HOP * STACK * 4 samples = 80 ms. */
#define GRANITE_SAMPLES_PER_FRAME \
    (GRANITE_HOP_LENGTH * GRANITE_STACK_FACTOR * 4)

/* ========================================================================
 * Weights
 * ======================================================================== */

/* All weights are bf16 in the checkpoint and stay mmap'd; the BatchNorm
 * running statistics are the only f32 tensors. */
typedef struct {
    const uint16_t *norm_ff1_w,  *norm_ff1_b;      /* [HIDDEN] LayerNorm */
    const uint16_t *ff1_l1_w,    *ff1_l1_b;        /* [4H, H] / [4H] */
    const uint16_t *ff1_l2_w,    *ff1_l2_b;        /* [H, 4H] / [H] */

    const uint16_t *norm_att_w,  *norm_att_b;
    const uint16_t *q_w, *k_w, *v_w;               /* [H, H], no bias */
    const uint16_t *o_w, *o_b;                     /* [H, H] / [H] */
    const uint16_t *rel_pos_emb;                   /* [2*MAX_POS_EMB+1, DIM_HEAD] */

    const uint16_t *norm_conv_w, *norm_conv_b;     /* LayerNorm over HIDDEN */
    const uint16_t *pw1_w, *pw1_b;                 /* [4H, H] / [4H] (GLU halves it) */
    const uint16_t *dw_w;                          /* [2H, 1, K] depthwise */
    const uint16_t *bn_w, *bn_b;                   /* BatchNorm affine [2H] */
    const float    *bn_mean, *bn_var;              /* BatchNorm stats [2H], f32 */
    const uint16_t *pw2_w, *pw2_b;                 /* [H, 2H] / [H] */

    const uint16_t *norm_ff2_w,  *norm_ff2_b;
    const uint16_t *ff2_l1_w,    *ff2_l1_b;
    const uint16_t *ff2_l2_w,    *ff2_l2_b;

    const uint16_t *norm_out_w,  *norm_out_b;      /* post-block LayerNorm */

    int subsample;                                  /* stride-2 depthwise conv */
} granite_layer_t;

typedef struct {
    const uint16_t *input_linear_w, *input_linear_b;  /* [H, 320] / [H] */
    granite_layer_t layers[GRANITE_NUM_LAYERS];
    const uint16_t *out_w, *out_b;                    /* [VOCAB, H] CTC head */
    const uint16_t *out_mid_w, *out_mid_b;            /* [H, VOCAB] mid-injection */
} granite_weights_t;

/* ========================================================================
 * Model handle
 * ======================================================================== */

struct granite_tokenizer;

typedef struct {
    multi_safetensors_t *st;           /* owns the mmap backing every weight */
    granite_weights_t w;
    struct granite_tokenizer *tok;

    /* Shaw relative-position index table [CONTEXT_SIZE * CONTEXT_SIZE]. */
    int32_t *attention_dists;

    /* Precomputed mel filterbank [N_MELS][N_FFT/2+1], sparse-friendly f32. */
    float *mel_filters;

    int verbose;
} granite_model_t;

/* ========================================================================
 * API
 * ======================================================================== */

/* Load model from a directory containing model.safetensors + tokenizer.json. */
granite_model_t *granite_load(const char *model_dir, int verbose);
void granite_free(granite_model_t *m);

/* Transcribe mono 16 kHz float samples in [-1, 1].
 * Returns a malloc'd UTF-8 string the caller must free, or NULL on error. */
char *granite_transcribe(granite_model_t *m, const float *samples, int n_samples);

/* Lower-level pieces, exposed for the parity harness (tools/parity.c).
 *
 * granite_frontend: samples -> [n_frames, INPUT_DIM] stacked log-mel + deltas.
 * granite_encode:   feats -> CTC logits [n_out_frames, VOCAB].
 * granite_ctc_greedy: logits -> collapsed ids (blanks and repeats removed). */
float *granite_frontend(const granite_model_t *m, const float *samples,
                        int n_samples, int *out_frames);
float *granite_encode(const granite_model_t *m, const float *feats,
                      int n_frames, int *out_frames);
int granite_ctc_greedy(const float *logits, int n_frames, int vocab, int *out_ids);

/* Resolve every weight pointer into the mmap'd checkpoint. Returns 0 on
 * success, -1 if any tensor is missing or has an unexpected dtype. */
int granite_bind_weights(granite_weights_t *w, const multi_safetensors_t *ms);

#endif /* GRANITE_H */
