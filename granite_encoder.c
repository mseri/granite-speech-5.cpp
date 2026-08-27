/*
 * granite_encoder.c - Conformer encoder forward pass + weight binding
 *
 * Per block (granite_encoder._ConformerBlock):
 *   x += 0.5 * FF1(x)
 *   x += Attention(x)            block-local, Shaw relative-position bias
 *   x  = ConvModule(x) + pool(x) subsampling blocks halve time; others add
 *   x += 0.5 * FF2(x)
 *   x  = LayerNorm(x)
 * After block 8 (num_layers / 2) the CTC head output is softmaxed and folded
 * back in through out_mid.
 */

#include "granite.h"
#include "granite_kernels.h"
#include "granite_safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FF_INNER   (GRANITE_HIDDEN * GRANITE_FF_MULT)      /* 4096 */
#define CONV_INNER (GRANITE_HIDDEN * GRANITE_CONV_EXPANSION) /* 2048 */
#define LN_EPS 1e-5f
#define BN_EPS 1e-5f

/* ========================================================================
 * Weight binding
 * ======================================================================== */

static const uint16_t *bind_bf16(const multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "granite: missing tensor %s\n", name);
        return NULL;
    }
    if (!safetensor_is_bf16(t)) {
        fprintf(stderr, "granite: expected bf16 for %s\n", name);
        return NULL;
    }
    return safetensors_get_bf16_direct(sf, t);
}

static const float *bind_f32(const multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "granite: missing tensor %s\n", name);
        return NULL;
    }
    return (const float *)safetensors_data(sf, t);
}

#define BIND(field, fmt, ...)                                        \
    do {                                                             \
        char _n[256];                                                \
        snprintf(_n, sizeof(_n), fmt, __VA_ARGS__);                  \
        if (!(field = bind_bf16(ms, _n))) return -1;                 \
    } while (0)

#define BIND_F32(field, fmt, ...)                                    \
    do {                                                             \
        char _n[256];                                                \
        snprintf(_n, sizeof(_n), fmt, __VA_ARGS__);                  \
        if (!(field = bind_f32(ms, _n))) return -1;                  \
    } while (0)

int granite_bind_weights(granite_weights_t *w, const multi_safetensors_t *ms) {
    if (!(w->input_linear_w = bind_bf16(ms, "encoder.input_linear.weight"))) return -1;
    if (!(w->input_linear_b = bind_bf16(ms, "encoder.input_linear.bias"))) return -1;
    if (!(w->out_w = bind_bf16(ms, "encoder.out.weight"))) return -1;
    if (!(w->out_b = bind_bf16(ms, "encoder.out.bias"))) return -1;
    if (!(w->out_mid_w = bind_bf16(ms, "encoder.out_mid.weight"))) return -1;
    if (!(w->out_mid_b = bind_bf16(ms, "encoder.out_mid.bias"))) return -1;

    for (int i = 0; i < GRANITE_NUM_LAYERS; i++) {
        granite_layer_t *L = &w->layers[i];
        L->subsample = (i < GRANITE_SUBSAMPLE_LAYERS);

        BIND(L->norm_ff1_w, "encoder.layers.%d.norm_feed_forward1.weight", i);
        BIND(L->norm_ff1_b, "encoder.layers.%d.norm_feed_forward1.bias", i);
        BIND(L->ff1_l1_w,   "encoder.layers.%d.feed_forward1.linear1.weight", i);
        BIND(L->ff1_l1_b,   "encoder.layers.%d.feed_forward1.linear1.bias", i);
        BIND(L->ff1_l2_w,   "encoder.layers.%d.feed_forward1.linear2.weight", i);
        BIND(L->ff1_l2_b,   "encoder.layers.%d.feed_forward1.linear2.bias", i);

        BIND(L->norm_att_w, "encoder.layers.%d.norm_self_att.weight", i);
        BIND(L->norm_att_b, "encoder.layers.%d.norm_self_att.bias", i);
        BIND(L->q_w, "encoder.layers.%d.self_attn.q_proj.weight", i);
        BIND(L->k_w, "encoder.layers.%d.self_attn.k_proj.weight", i);
        BIND(L->v_w, "encoder.layers.%d.self_attn.v_proj.weight", i);
        BIND(L->o_w, "encoder.layers.%d.self_attn.o_proj.weight", i);
        BIND(L->o_b, "encoder.layers.%d.self_attn.o_proj.bias", i);
        BIND(L->rel_pos_emb, "encoder.layers.%d.self_attn.rel_pos_emb.weight", i);

        BIND(L->norm_conv_w, "encoder.layers.%d.norm_conv.weight", i);
        BIND(L->norm_conv_b, "encoder.layers.%d.norm_conv.bias", i);
        BIND(L->pw1_w, "encoder.layers.%d.conv.pointwise_lin1.weight", i);
        BIND(L->pw1_b, "encoder.layers.%d.conv.pointwise_lin1.bias", i);
        BIND(L->dw_w,  "encoder.layers.%d.conv.depthwise_conv.weight", i);
        BIND(L->bn_w,  "encoder.layers.%d.conv.norm.weight", i);
        BIND(L->bn_b,  "encoder.layers.%d.conv.norm.bias", i);
        BIND_F32(L->bn_mean, "encoder.layers.%d.conv.norm.running_mean", i);
        BIND_F32(L->bn_var,  "encoder.layers.%d.conv.norm.running_var", i);
        BIND(L->pw2_w, "encoder.layers.%d.conv.pointwise_lin2.weight", i);
        BIND(L->pw2_b, "encoder.layers.%d.conv.pointwise_lin2.bias", i);

        BIND(L->norm_ff2_w, "encoder.layers.%d.norm_feed_forward2.weight", i);
        BIND(L->norm_ff2_b, "encoder.layers.%d.norm_feed_forward2.bias", i);
        BIND(L->ff2_l1_w,   "encoder.layers.%d.feed_forward2.linear1.weight", i);
        BIND(L->ff2_l1_b,   "encoder.layers.%d.feed_forward2.linear1.bias", i);
        BIND(L->ff2_l2_w,   "encoder.layers.%d.feed_forward2.linear2.weight", i);
        BIND(L->ff2_l2_b,   "encoder.layers.%d.feed_forward2.linear2.bias", i);

        BIND(L->norm_out_w, "encoder.layers.%d.norm_out.weight", i);
        BIND(L->norm_out_b, "encoder.layers.%d.norm_out.bias", i);
    }
    return 0;
}

/* ========================================================================
 * Scratch buffers
 * ======================================================================== */

typedef struct {
    float *norm;    /* [T, HIDDEN]     normalized input to a sublayer */
    float *ff;      /* [T, FF_INNER]   feed-forward / pointwise_lin1 inner */
    float *q, *k, *v, *attn;  /* [T, HIDDEN] */
    float *conv;    /* [T, CONV_INNER] post-GLU conv activations */
    float *conv_o;  /* [T, CONV_INNER] depthwise conv output */
    float *proj;    /* [T, HIDDEN]     sublayer output before the residual add */
    float *mid;     /* [T, VOCAB]      mid-injection logits */
} scratch_t;

static void scratch_free(scratch_t *s) {
    free(s->norm); free(s->ff); free(s->q); free(s->k); free(s->v);
    free(s->attn); free(s->conv); free(s->conv_o); free(s->proj); free(s->mid);
    memset(s, 0, sizeof(*s));
}

static int scratch_alloc(scratch_t *s, int T) {
    memset(s, 0, sizeof(*s));
    size_t h = (size_t)T * GRANITE_HIDDEN * sizeof(float);
    s->norm   = malloc(h);
    s->q      = malloc(h);
    s->k      = malloc(h);
    s->v      = malloc(h);
    s->attn   = malloc(h);
    s->proj   = malloc(h);
    s->ff     = malloc((size_t)T * FF_INNER * sizeof(float));
    s->conv   = malloc((size_t)T * CONV_INNER * sizeof(float));
    s->conv_o = malloc((size_t)T * CONV_INNER * sizeof(float));
    /* Mid-injection runs at layer 8, well past both subsampling blocks, so it
     * only ever sees T/4 frames. Sizing this at T would reserve 4x the largest
     * buffer in the model (T * 16384 floats). */
    s->mid    = malloc(((size_t)(T / 4) + 2) * GRANITE_VOCAB * sizeof(float));
    if (!s->norm || !s->q || !s->k || !s->v || !s->attn || !s->proj ||
        !s->ff || !s->conv || !s->conv_o || !s->mid) {
        scratch_free(s);
        return -1;
    }
    return 0;
}

/* ========================================================================
 * Sublayers
 * ======================================================================== */

/* x += 0.5 * FF(x) */
static void feed_forward(float *x, int T, scratch_t *s,
                         const uint16_t *nw, const uint16_t *nb,
                         const uint16_t *l1w, const uint16_t *l1b,
                         const uint16_t *l2w, const uint16_t *l2b) {
    granite_layer_norm_bf16(s->norm, x, nw, nb, T, GRANITE_HIDDEN, LN_EPS);
    granite_linear_bf16(s->ff, s->norm, l1w, l1b, T, GRANITE_HIDDEN, FF_INNER);
    granite_silu(s->ff, T * FF_INNER);
    granite_linear_bf16(s->proj, s->ff, l2w, l2b, T, FF_INNER, GRANITE_HIDDEN);

    int n = T * GRANITE_HIDDEN;
    for (int i = 0; i < n; i++) x[i] += 0.5f * s->proj[i];
}

/* x += Attention(x). Full blocks of CONTEXT_SIZE, then a trailing block at its
 * true length (granite_encoder._SeparateQKVAttention). */
static void attention(float *x, int T, scratch_t *s, const granite_layer_t *L,
                      const int32_t *dists) {
    const int c = GRANITE_CONTEXT_SIZE;
    const int inner = GRANITE_NUM_HEADS * GRANITE_DIM_HEAD;
    const float scale = 1.0f / sqrtf((float)GRANITE_DIM_HEAD);

    granite_layer_norm_bf16(s->norm, x, L->norm_att_w, L->norm_att_b,
                            T, GRANITE_HIDDEN, LN_EPS);
    granite_linear_bf16(s->q, s->norm, L->q_w, NULL, T, GRANITE_HIDDEN, inner);
    granite_linear_bf16(s->k, s->norm, L->k_w, NULL, T, GRANITE_HIDDEN, inner);
    granite_linear_bf16(s->v, s->norm, L->v_w, NULL, T, GRANITE_HIDDEN, inner);

    int nb_full = T / c;
    int nr = T % c;
    if (nb_full > 0) {
        granite_block_attention(s->attn, s->q, s->k, s->v, nb_full * c, c,
                                GRANITE_NUM_HEADS, GRANITE_DIM_HEAD, scale,
                                L->rel_pos_emb, dists, c);
    }
    if (nr > 0) {
        size_t off = (size_t)nb_full * c * inner;
        granite_block_attention(s->attn + off, s->q + off, s->k + off,
                                s->v + off, nr, nr,
                                GRANITE_NUM_HEADS, GRANITE_DIM_HEAD, scale,
                                L->rel_pos_emb, dists, c);
    }

    granite_linear_bf16(s->proj, s->attn, L->o_w, L->o_b, T, inner, GRANITE_HIDDEN);

    int n = T * GRANITE_HIDDEN;
    for (int i = 0; i < n; i++) x[i] += s->proj[i];
}

/* Conv module. Writes the module output to s->proj and returns its frame count
 * (T for stride 1, T/2 for the subsampling blocks). */
static int conv_module(const float *x, int T, scratch_t *s,
                       const granite_layer_t *L) {
    const int k = GRANITE_CONV_KERNEL;
    const int stride = L->subsample ? 2 : 1;
    /* Stock GraniteSpeechConformerDepthWiseConv1d pads (pad, pad - (k+1) % 2);
     * k = 7 is odd, so both sides get k/2 = 3. */
    const int pad = k / 2;
    const int pad_r = pad - (k + 1) % 2;

    granite_layer_norm_bf16(s->norm, x, L->norm_conv_w, L->norm_conv_b,
                            T, GRANITE_HIDDEN, LN_EPS);
    granite_linear_bf16(s->ff, s->norm, L->pw1_w, L->pw1_b,
                        T, GRANITE_HIDDEN, CONV_INNER * 2);
    granite_glu(s->conv, s->ff, T, CONV_INNER);

    granite_depthwise_conv1d_bf16(s->conv_o, s->conv, L->dw_w, T, CONV_INNER,
                                  k, stride, pad, pad_r);
    int out_T = (T + pad + pad_r - k) / stride + 1;

    granite_batch_norm_bf16(s->conv_o, L->bn_w, L->bn_b, L->bn_mean, L->bn_var,
                            out_T, CONV_INNER, BN_EPS);
    granite_silu(s->conv_o, out_T * CONV_INNER);
    granite_linear_bf16(s->proj, s->conv_o, L->pw2_w, L->pw2_b,
                        out_T, CONV_INNER, GRANITE_HIDDEN);
    return out_T;
}

/* ========================================================================
 * Forward
 * ======================================================================== */

float *granite_encode(const granite_model_t *m, const float *feats,
                      int n_frames, int *out_frames) {
    int T = n_frames;
    if (T <= 0) return NULL;

    scratch_t s;
    if (scratch_alloc(&s, T) != 0) return NULL;

    float *x = malloc((size_t)T * GRANITE_HIDDEN * sizeof(float));
    if (!x) { scratch_free(&s); return NULL; }

    granite_linear_bf16(x, feats, m->w.input_linear_w, m->w.input_linear_b,
                        T, GRANITE_INPUT_DIM, GRANITE_HIDDEN);

    for (int i = 0; i < GRANITE_NUM_LAYERS; i++) {
        const granite_layer_t *L = &m->w.layers[i];

        feed_forward(x, T, &s, L->norm_ff1_w, L->norm_ff1_b,
                     L->ff1_l1_w, L->ff1_l1_b, L->ff1_l2_w, L->ff1_l2_b);
        attention(x, T, &s, L, m->attention_dists);

        int conv_T = conv_module(x, T, &s, L);
        if (L->subsample) {
            /* Mean-pool the residual to half rate (dropping a trailing odd
             * frame) and trim the conv output to match. */
            int t_half = T / 2;
            for (int t = 0; t < t_half; t++) {
                float *dst = x + (size_t)t * GRANITE_HIDDEN;
                const float *a = x + (size_t)(2 * t) * GRANITE_HIDDEN;
                const float *b = a + GRANITE_HIDDEN;
                const float *c = s.proj + (size_t)t * GRANITE_HIDDEN;
                for (int j = 0; j < GRANITE_HIDDEN; j++)
                    dst[j] = 0.5f * (a[j] + b[j]) + c[j];
            }
            T = t_half;
            (void)conv_T;
        } else {
            granite_add_inplace(x, s.proj, T * GRANITE_HIDDEN);
        }

        feed_forward(x, T, &s, L->norm_ff2_w, L->norm_ff2_b,
                     L->ff2_l1_w, L->ff2_l1_b, L->ff2_l2_w, L->ff2_l2_b);
        granite_layer_norm_bf16(s.norm, x, L->norm_out_w, L->norm_out_b,
                                T, GRANITE_HIDDEN, LN_EPS);
        memcpy(x, s.norm, (size_t)T * GRANITE_HIDDEN * sizeof(float));

        /* Mid-injection after block num_layers/2 (1-based index). */
        if (i + 1 == GRANITE_NUM_LAYERS / 2) {
            granite_linear_bf16(s.mid, x, m->w.out_w, m->w.out_b,
                                T, GRANITE_HIDDEN, GRANITE_VOCAB);
            granite_softmax_rows(s.mid, T, GRANITE_VOCAB);
            granite_linear_bf16(s.proj, s.mid, m->w.out_mid_w, m->w.out_mid_b,
                                T, GRANITE_VOCAB, GRANITE_HIDDEN);
            granite_add_inplace(x, s.proj, T * GRANITE_HIDDEN);
        }
    }

    float *logits = malloc((size_t)T * GRANITE_VOCAB * sizeof(float));
    if (logits)
        granite_linear_bf16(logits, x, m->w.out_w, m->w.out_b,
                            T, GRANITE_HIDDEN, GRANITE_VOCAB);

    free(x);
    scratch_free(&s);
    *out_frames = T;
    return logits;
}

int granite_ctc_greedy(const float *logits, int n_frames, int vocab, int *out_ids) {
    int n = 0, prev = -1;
    for (int t = 0; t < n_frames; t++) {
        const float *row = logits + (size_t)t * vocab;
        int best = 0;
        float best_v = row[0];
        for (int i = 1; i < vocab; i++) {
            if (row[i] > best_v) { best_v = row[i]; best = i; }
        }
        if (best != prev && best != GRANITE_BLANK_ID) out_ids[n++] = best;
        prev = best;
    }
    return n;
}
