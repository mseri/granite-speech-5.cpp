/*
 * granite.c - Model load/free and the end-to-end transcription flow
 */

#include "granite.h"
#include "granite_audio.h"
#include "granite_kernels.h"
#include "granite_safetensors.h"
#include "granite_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Shaw relative-position index table, matching GraniteSpeechCTCEncoder:
 *   dists[i][j] = clamp(i - j, -context, context) + max_pos_emb
 * With context_size 128 the clamp never binds; the offset centres the range on
 * the 1025-row embedding table. */
static int32_t *build_attention_dists(void) {
    const int c = GRANITE_CONTEXT_SIZE;
    int32_t *d = malloc((size_t)c * c * sizeof(int32_t));
    if (!d) return NULL;
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < c; j++) {
            int rel = i - j;
            if (rel < -c) rel = -c;
            if (rel > c) rel = c;
            d[(size_t)i * c + j] = rel + GRANITE_MAX_POS_EMB;
        }
    }
    return d;
}

granite_model_t *granite_load(const char *model_dir, int verbose) {
    granite_model_t *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->verbose = verbose;

    m->st = multi_safetensors_open(model_dir);
    if (!m->st) {
        fprintf(stderr, "granite: failed to open safetensors in %s\n", model_dir);
        granite_free(m);
        return NULL;
    }

    if (granite_bind_weights(&m->w, m->st) != 0) {
        granite_free(m);
        return NULL;
    }

    char tok_path[1024];
    snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.json", model_dir);
    m->tok = granite_tokenizer_load(tok_path);
    if (!m->tok) {
        granite_free(m);
        return NULL;
    }

    m->attention_dists = build_attention_dists();
    m->mel_filters = granite_mel_filterbank(GRANITE_SAMPLE_RATE, GRANITE_N_FFT,
                                            GRANITE_N_MELS);
    if (!m->attention_dists || !m->mel_filters) {
        granite_free(m);
        return NULL;
    }

    if (verbose)
        fprintf(stderr, "granite: loaded %d conformer layers from %s\n",
                GRANITE_NUM_LAYERS, model_dir);
    return m;
}

void granite_free(granite_model_t *m) {
    if (!m) return;
    if (m->st) multi_safetensors_close(m->st);
    if (m->tok) granite_tokenizer_free(m->tok);
    free(m->attention_dists);
    free(m->mel_filters);
    free(m);
}

char *granite_transcribe(granite_model_t *m, const float *samples, int n_samples) {
    if (!m || !samples || n_samples <= 0) return NULL;

    int n_frames = 0;
    float *feats = granite_frontend(m, samples, n_samples, &n_frames);
    if (!feats) return NULL;
    if (m->verbose)
        fprintf(stderr, "granite: %d samples -> %d input frames\n",
                n_samples, n_frames);

    int out_frames = 0;
    float *logits = granite_encode(m, feats, n_frames, &out_frames);
    free(feats);
    if (!logits) return NULL;

    int *ids = malloc((size_t)out_frames * sizeof(int));
    if (!ids) { free(logits); return NULL; }
    int n_ids = granite_ctc_greedy(logits, out_frames, GRANITE_VOCAB, ids);
    free(logits);

    if (m->verbose)
        fprintf(stderr, "granite: %d encoder frames -> %d tokens\n",
                out_frames, n_ids);

    char *text = granite_tokenizer_decode(m->tok, ids, n_ids);
    free(ids);
    return text;
}
