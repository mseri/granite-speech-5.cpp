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

void granite_default_params(granite_params_t *p) {
    p->segment_sec = 0.0f;              /* whole-clip decode */
    p->cut_window_sec = 3.0f;
    p->stream_chunk_sec = 2.0f;
    /* Two attention blocks of context, so a committed frame has seen a full
     * block behind it even at the window's left edge. */
    p->stream_window_sec = 2.0f * GRANITE_BLOCK_SECONDS;
    p->stream_lookahead_sec = 1.6f;
}

/* ========================================================================
 * Growable output string
 * ======================================================================== */

typedef struct {
    char *buf;
    size_t len, cap;
} strbuf_t;

static int sb_append(strbuf_t *s, const char *text) {
    size_t n = strlen(text);
    if (s->len + n + 1 > s->cap) {
        size_t cap = s->cap ? s->cap : 256;
        while (s->len + n + 1 > cap) cap *= 2;
        char *nb = realloc(s->buf, cap);
        if (!nb) return -1;
        s->buf = nb;
        s->cap = cap;
    }
    memcpy(s->buf + s->len, text, n + 1);
    s->len += n;
    return 0;
}

/* ========================================================================
 * Segmented mode
 * ======================================================================== */

/* Pick a cut near `target`, preferring the quietest 20 ms window within
 * +/- `radius` samples, so segment boundaries land in pauses rather than
 * mid-word. Returns a sample index in (start, n_samples]. */
static int find_cut_point(const float *samples, int n_samples, int start,
                          int target, int radius) {
    const int win = GRANITE_SAMPLE_RATE / 50;   /* 20 ms */
    int lo = target - radius;
    int hi = target + radius;
    if (lo < start + win) lo = start + win;
    if (hi > n_samples - win) hi = n_samples - win;
    if (lo >= hi) return target < n_samples ? target : n_samples;

    /* Step by a quarter window; sub-sample energy for speed. */
    int step = win / 4 > 0 ? win / 4 : 1;
    int best = target;
    double best_e = -1.0;
    for (int c = lo; c <= hi; c += step) {
        double e = 0.0;
        for (int i = c - win / 2; i < c + win / 2; i += 4) {
            float v = samples[i];
            e += (double)v * v;
        }
        if (best_e < 0.0 || e < best_e) {
            best_e = e;
            best = c;
        }
    }
    return best;
}

char *granite_transcribe_segmented(granite_model_t *m, const float *samples,
                                   int n_samples, const granite_params_t *p,
                                   granite_text_cb cb, void *user) {
    if (!m || !samples || n_samples <= 0) return NULL;

    granite_params_t defaults;
    if (!p) { granite_default_params(&defaults); p = &defaults; }

    int seg = (int)(p->segment_sec * GRANITE_SAMPLE_RATE);
    if (seg <= 0 || seg >= n_samples) {
        char *text = granite_transcribe(m, samples, n_samples);
        if (text && cb) cb(text, user);
        return text;
    }
    /* Segments shorter than one attention block lose context the model expects. */
    int min_seg = (int)(GRANITE_BLOCK_SECONDS * GRANITE_SAMPLE_RATE);
    if (seg < min_seg) {
        if (m->verbose)
            fprintf(stderr, "granite: raising segment length to %.1fs "
                            "(one attention block)\n", GRANITE_BLOCK_SECONDS);
        seg = min_seg;
    }
    int radius = (int)(p->cut_window_sec * GRANITE_SAMPLE_RATE);

    strbuf_t out = { 0 };
    int pos = 0, index = 0;
    while (pos < n_samples) {
        int end = pos + seg;
        if (end >= n_samples) {
            end = n_samples;
        } else {
            end = find_cut_point(samples, n_samples, pos, end, radius);
        }

        char *piece = granite_transcribe(m, samples + pos, end - pos);
        if (piece && *piece) {
            if (out.len > 0 && sb_append(&out, " ") != 0) { free(piece); break; }
            size_t before = out.len;
            if (sb_append(&out, piece) != 0) { free(piece); break; }
            if (cb) cb(out.buf + before, user);
        }
        free(piece);

        if (m->verbose)
            fprintf(stderr, "granite: segment %d [%.2fs, %.2fs]\n", index++,
                    (double)pos / GRANITE_SAMPLE_RATE,
                    (double)end / GRANITE_SAMPLE_RATE);
        pos = end;
    }
    return out.buf;
}

/* ========================================================================
 * Streaming mode
 * ======================================================================== */

/* Per-frame argmax over the CTC vocabulary. */
static void argmax_frames(const float *logits, int n_frames, int *out) {
    for (int t = 0; t < n_frames; t++) {
        const float *row = logits + (size_t)t * GRANITE_VOCAB;
        int best = 0;
        float best_v = row[0];
        for (int i = 1; i < GRANITE_VOCAB; i++)
            if (row[i] > best_v) { best_v = row[i]; best = i; }
        out[t] = best;
    }
}

char *granite_transcribe_stream(granite_model_t *m, granite_live_audio_t *la,
                                const granite_params_t *p,
                                granite_text_cb cb, void *user) {
    if (!m || !la) return NULL;

    granite_params_t defaults;
    if (!p) { granite_default_params(&defaults); p = &defaults; }

    const int spf = GRANITE_SAMPLES_PER_FRAME;
    int chunk = (int)(p->stream_chunk_sec * GRANITE_SAMPLE_RATE);
    int window = (int)(p->stream_window_sec * GRANITE_SAMPLE_RATE);
    int lookahead_frames = (int)(p->stream_lookahead_sec *
                                 GRANITE_SAMPLE_RATE / spf);
    if (chunk < spf) chunk = spf;
    if (window < chunk) window = chunk;

    /* Per-frame argmax over the whole stream, indexed by absolute frame.
     * Frames behind the lookahead are final; the rest get overwritten as the
     * window slides, which is what keeps committed text stable. */
    int *raw = NULL;
    int raw_cap = 0, committed = 0;

    /* Block-aligning the window start rounds it down, so the span can exceed
     * `window` by up to one block. Size the buffers for that. */
    int win_max = window + GRANITE_CONTEXT_SIZE * spf;
    float *win_buf = malloc((size_t)win_max * sizeof(float));
    int *ids = malloc((size_t)(win_max / spf + 8) * sizeof(int));
    if (!win_buf || !ids) { free(win_buf); free(ids); return NULL; }

    strbuf_t out = { 0 };
    size_t emitted = 0;
    int win_start = 0, want = chunk, eof = 0;

    int win_end = 0;
    for (;;) {
        int avail = granite_live_audio_wait(la, want, &eof);

        /* Advance the window end to whatever has arrived, but never far enough
         * that its start would pass the committed frontier -- otherwise a fast
         * producer (a pipe delivering the whole file at once) would leave the
         * skipped span undecoded. With a backlog this strides window-at-a-time. */
        int max_end = committed * spf + window;
        int end = avail < max_end ? avail : max_end;
        if (end <= win_end) {
            /* No new audio. If more may still arrive, go back and wait; at EOF
             * fall through once more so the trailing lookahead is released and
             * the final frames get committed. The bottom-of-loop check then
             * ends the loop, so this runs at most once. */
            if (!eof) break;
        }
        win_end = end;

        /* Align the window start to an attention-block boundary, not merely a
         * frame boundary. Attention is block-local over GRANITE_CONTEXT_SIZE
         * frames, so a start that is not block-aligned shifts the whole block
         * grid relative to the previous window and re-decodes frames under
         * different context -- which is what produces boundary artifacts. */
        const int block_samples = GRANITE_CONTEXT_SIZE * spf;
        win_start = end - window;
        if (win_start < 0) win_start = 0;
        win_start = (win_start / block_samples) * block_samples;

        int win_n = end - win_start;
        granite_live_audio_copy(la, win_start, win_n, win_buf);

        int n_frames = 0;
        float *feats = granite_frontend(m, win_buf, win_n, &n_frames);
        if (!feats) break;
        int enc_frames = 0;
        float *logits = granite_encode(m, feats, n_frames, &enc_frames);
        free(feats);
        if (!logits) break;

        int base = win_start / spf;
        if (base + enc_frames > raw_cap) {
            int cap = raw_cap ? raw_cap : 1024;
            while (base + enc_frames > cap) cap *= 2;
            int *nr = realloc(raw, (size_t)cap * sizeof(int));
            if (!nr) { free(logits); break; }
            raw = nr;
            raw_cap = cap;
        }
        argmax_frames(logits, enc_frames, ids);
        free(logits);
        /* Only write frames that are not yet committed. The window overlaps
         * already-committed frames and re-decodes them under different context;
         * letting those results land would mutate the committed prefix, so the
         * collapsed text would stop being a pure extension of what was already
         * emitted and the delta below would splice in duplicated fragments. */
        for (int i = 0; i < enc_frames; i++) {
            int f = base + i;
            if (f >= committed) raw[f] = ids[i];
        }

        /* Everything except the trailing lookahead is now stable. The lookahead
         * only applies while more audio may still arrive for this span. */
        int limit = base + enc_frames;
        if (!(eof && win_end >= avail)) {
            limit -= lookahead_frames;
            /* Before a full attention block of audio exists, no frame has seen
             * the context the model was trained with -- the window's right edge
             * is padding, not signal. Committing there bakes in artifacts that
             * later windows cannot undo, so hold everything back. Clips shorter
             * than a block therefore commit once, at EOF, matching offline. */
            if (avail < GRANITE_CONTEXT_SIZE * spf) limit = 0;
        }
        if (limit > committed) committed = limit;
        if (committed < 0) committed = 0;

        /* CTC-collapse the committed prefix. Since committed frames never
         * change, this only ever extends the previous decode. */
        int *collapsed = malloc((size_t)committed * sizeof(int));
        if (!collapsed) break;
        int n_ids = 0, prev = -1;
        for (int i = 0; i < committed; i++) {
            if (raw[i] != prev && raw[i] != GRANITE_BLANK_ID)
                collapsed[n_ids++] = raw[i];
            prev = raw[i];
        }
        char *text = granite_tokenizer_decode(m->tok, collapsed, n_ids);
        free(collapsed);
        if (!text) break;

        out.len = 0;
        if (out.buf) out.buf[0] = '\0';
        if (sb_append(&out, text) != 0) { free(text); break; }
        free(text);

        if (cb && out.len > emitted) {
            cb(out.buf + emitted, user);
            emitted = out.len;
        }

        /* Done only once the window has actually reached the end of the stream;
         * at EOF with a backlog there are still earlier spans to decode. */
        if (eof && win_end >= avail) break;
        want = win_end + chunk;
    }

    free(win_buf);
    free(ids);
    free(raw);
    return out.buf ? out.buf : calloc(1, 1);
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
