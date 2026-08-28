/* Audio loading and the Granite Speech 5.0 log-mel feature extractor. */

#include "granite_audio.h"
#include "granite.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t rd_u16(const uint8_t *p) {
    return (uint16_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8));
}

/* Linear-interpolation resampler; 16 kHz input takes the copy path. */
static float *resample_linear(const float *in, int n_in, int sr_in, int sr_out,
                              int *out_n) {
    if (sr_in == sr_out) {
        float *copy = malloc((size_t)n_in * sizeof(float));
        if (!copy) return NULL;
        memcpy(copy, in, (size_t)n_in * sizeof(float));
        *out_n = n_in;
        return copy;
    }
    double ratio = (double)sr_out / sr_in;
    int n_out = (int)(n_in * ratio);
    if (n_out < 1) return NULL;
    float *out = malloc((size_t)n_out * sizeof(float));
    if (!out) return NULL;
    for (int i = 0; i < n_out; i++) {
        double src = i / ratio;
        int i0 = (int)src;
        int i1 = i0 + 1 < n_in ? i0 + 1 : n_in - 1;
        float frac = (float)(src - i0);
        out[i] = in[i0] * (1.0f - frac) + in[i1] * frac;
    }
    *out_n = n_out;
    return out;
}

float *granite_parse_wav_buffer(const uint8_t *data, size_t size, int *out_n_samples) {
    if (size < 44 || memcmp(data, "RIFF", 4) != 0 || memcmp(data + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "granite: not a RIFF/WAVE file\n");
        return NULL;
    }

    int channels = 0, sample_rate = 0, bits = 0;
    const uint8_t *pcm = NULL;
    size_t pcm_bytes = 0;

    size_t pos = 12;
    while (pos + 8 <= size) {
        const uint8_t *id = data + pos;
        uint32_t clen = rd_u32(data + pos + 4);
        const uint8_t *body = data + pos + 8;
        if (pos + 8 + clen > size) clen = (uint32_t)(size - pos - 8);

        if (memcmp(id, "fmt ", 4) == 0 && clen >= 16) {
            channels = rd_u16(body + 2);
            sample_rate = (int)rd_u32(body + 4);
            bits = rd_u16(body + 14);
        } else if (memcmp(id, "data", 4) == 0) {
            pcm = body;
            pcm_bytes = clen;
        }
        pos += 8 + clen + (clen & 1);   /* chunks are word-aligned */
    }

    if (!pcm || channels < 1 || sample_rate < 1) {
        fprintf(stderr, "granite: malformed WAV (no fmt/data chunk)\n");
        return NULL;
    }
    if (bits != 16) {
        fprintf(stderr, "granite: only 16-bit PCM WAV is supported (got %d-bit)\n", bits);
        return NULL;
    }

    int n_frames = (int)(pcm_bytes / (size_t)(2 * channels));
    float *mono = malloc((size_t)n_frames * sizeof(float));
    if (!mono) return NULL;
    for (int i = 0; i < n_frames; i++) {
        int acc = 0;
        for (int c = 0; c < channels; c++) {
            int16_t s = (int16_t)rd_u16(pcm + (size_t)(i * channels + c) * 2);
            acc += s;
        }
        mono[i] = (float)acc / (channels * 32768.0f);
    }

    float *out = resample_linear(mono, n_frames, sample_rate,
                                 GRANITE_SAMPLE_RATE, out_n_samples);
    free(mono);
    return out;
}

float *granite_load_wav(const char *path, int *out_n_samples) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "granite: cannot open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) { fclose(f); return NULL; }

    uint8_t *buf = malloc((size_t)size);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);

    float *out = granite_parse_wav_buffer(buf, (size_t)size, out_n_samples);
    free(buf);
    return out;
}

float *granite_read_stdin(int *out_n_samples) {
    size_t cap = 1 << 20, len = 0;
    uint8_t *buf = malloc(cap);
    if (!buf) return NULL;
    for (;;) {
        if (len == cap) {
            cap *= 2;
            uint8_t *nb = realloc(buf, cap);
            if (!nb) { free(buf); return NULL; }
            buf = nb;
        }
        size_t got = fread(buf + len, 1, cap - len, stdin);
        if (got == 0) break;
        len += got;
    }

    float *out;
    if (len >= 12 && memcmp(buf, "RIFF", 4) == 0) {
        out = granite_parse_wav_buffer(buf, len, out_n_samples);
    } else {
        /* Raw s16le, 16 kHz, mono. */
        int n = (int)(len / 2);
        out = malloc((size_t)n * sizeof(float));
        if (out) {
            for (int i = 0; i < n; i++)
                out[i] = (float)(int16_t)rd_u16(buf + (size_t)i * 2) / 32768.0f;
            *out_n_samples = n;
        }
    }
    free(buf);
    return out;
}


struct granite_live_audio {
    float *buf;
    int len, cap;
    int eof;
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
};

static void live_append(granite_live_audio_t *la, const float *src, int n) {
    pthread_mutex_lock(&la->mutex);
    if (la->len + n > la->cap) {
        while (la->len + n > la->cap) la->cap = la->cap ? la->cap * 2 : 1 << 18;
        la->buf = realloc(la->buf, (size_t)la->cap * sizeof(float));
    }
    if (la->buf) {
        memcpy(la->buf + la->len, src, (size_t)n * sizeof(float));
        la->len += n;
    }
    pthread_cond_broadcast(&la->cond);
    pthread_mutex_unlock(&la->mutex);
}

/* Read up to n bytes. */
static size_t read_full(uint8_t *dst, size_t n) {
    size_t got = 0;
    while (got < n) {
        size_t r = fread(dst + got, 1, n - got, stdin);
        if (r == 0) break;
        got += r;
    }
    return got;
}

/* Consume a WAV header, or preserve the leading bytes when input is raw audio. */
static int consume_wav_header(uint8_t *lead, size_t *lead_n) {
    uint8_t riff[12];
    size_t got = read_full(riff, sizeof(riff));
    if (got < 12 || memcmp(riff, "RIFF", 4) != 0 || memcmp(riff + 8, "WAVE", 4) != 0) {
        memcpy(lead, riff, got);
        *lead_n = got;
        return 0;
    }

    for (;;) {
        uint8_t ch[8];
        if (read_full(ch, 8) < 8) break;
        uint32_t clen = rd_u32(ch + 4);

        if (memcmp(ch, "data", 4) == 0) break;

        if (memcmp(ch, "fmt ", 4) == 0 && clen >= 16) {
            uint8_t fmt[64];
            uint32_t take = clen < sizeof(fmt) ? clen : (uint32_t)sizeof(fmt);
            if (read_full(fmt, take) < take) break;
            int rate = (int)rd_u32(fmt + 4);
            int bits = rd_u16(fmt + 14);
            int channels = rd_u16(fmt + 2);
            if (rate != GRANITE_SAMPLE_RATE || bits != 16 || channels != 1)
                fprintf(stderr, "granite: --stream expects 16 kHz mono 16-bit "
                                "(got %d Hz, %d-bit, %d ch)\n", rate, bits, channels);
            clen -= take;
        }
        /* WAV chunks are padded to an even byte boundary. */
        uint32_t skip = clen + (clen & 1);
        uint8_t sink[4096];
        while (skip > 0) {
            uint32_t take = skip < sizeof(sink) ? skip : (uint32_t)sizeof(sink);
            if (read_full(sink, take) < take) return 1;
            skip -= take;
        }
    }
    *lead_n = 0;
    return 1;
}

static void *live_reader(void *arg) {
    granite_live_audio_t *la = arg;

    /* A WAV header is consumed if present; anything else is raw s16le. */
    uint8_t lead[12];
    size_t lead_n = 0;
    consume_wav_header(lead, &lead_n);
    if (lead_n >= 2) {
        int n = (int)(lead_n / 2);
        float tmp[6];
        for (int i = 0; i < n; i++)
            tmp[i] = (float)(int16_t)rd_u16(lead + (size_t)i * 2) / 32768.0f;
        live_append(la, tmp, n);
    }

    uint8_t chunk[8192];
    float conv[4096];
    int carry_valid = 0;
    uint8_t carry = 0;
    for (;;) {
        size_t r = fread(chunk, 1, sizeof(chunk), stdin);
        if (r == 0) break;

        size_t off = 0;
        int n = 0;
        /* A read can split a 16-bit sample; carry the odd byte over. */
        if (carry_valid) {
            uint8_t pair[2] = { carry, chunk[0] };
            conv[n++] = (float)(int16_t)rd_u16(pair) / 32768.0f;
            off = 1;
            carry_valid = 0;
        }
        for (; off + 2 <= r; off += 2) {
            conv[n++] = (float)(int16_t)rd_u16(chunk + off) / 32768.0f;
            if (n == (int)(sizeof(conv) / sizeof(conv[0]))) {
                live_append(la, conv, n);
                n = 0;
            }
        }
        if (off < r) { carry = chunk[off]; carry_valid = 1; }
        if (n > 0) live_append(la, conv, n);
    }

    pthread_mutex_lock(&la->mutex);
    la->eof = 1;
    pthread_cond_broadcast(&la->cond);
    pthread_mutex_unlock(&la->mutex);
    return NULL;
}

granite_live_audio_t *granite_live_audio_start_stdin(void) {
    granite_live_audio_t *la = calloc(1, sizeof(*la));
    if (!la) return NULL;
    pthread_mutex_init(&la->mutex, NULL);
    pthread_cond_init(&la->cond, NULL);
    if (pthread_create(&la->thread, NULL, live_reader, la) != 0) {
        free(la);
        return NULL;
    }
    return la;
}

void granite_live_audio_free(granite_live_audio_t *la) {
    if (!la) return;
    pthread_join(la->thread, NULL);
    pthread_mutex_destroy(&la->mutex);
    pthread_cond_destroy(&la->cond);
    free(la->buf);
    free(la);
}

int granite_live_audio_wait(granite_live_audio_t *la, int want, int *eof) {
    pthread_mutex_lock(&la->mutex);
    while (la->len < want && !la->eof)
        pthread_cond_wait(&la->cond, &la->mutex);
    int len = la->len;
    if (eof) *eof = la->eof;
    pthread_mutex_unlock(&la->mutex);
    return len;
}

void granite_live_audio_copy(granite_live_audio_t *la, int start, int n, float *dst) {
    pthread_mutex_lock(&la->mutex);
    if (start < 0) start = 0;
    if (start + n > la->len) n = la->len - start;
    if (n > 0) memcpy(dst, la->buf + start, (size_t)n * sizeof(float));
    pthread_mutex_unlock(&la->mutex);
}


static double hz_to_mel_htk(double f) {
    return 2595.0 * log10(1.0 + f / 700.0);
}

static double mel_to_hz_htk(double m) {
    return 700.0 * (pow(10.0, m / 2595.0) - 1.0);
}

float *granite_mel_filterbank(int sample_rate, int n_fft, int n_mels) {
    int n_freqs = n_fft / 2 + 1;
    float *fb = calloc((size_t)n_mels * n_freqs, sizeof(float));
    if (!fb) return NULL;

    /* Match torchaudio's evenly spaced frequency centres. */
    double f_max = sample_rate / 2;
    double *all_freqs = malloc((size_t)n_freqs * sizeof(double));
    double *f_pts = malloc((size_t)(n_mels + 2) * sizeof(double));
    if (!all_freqs || !f_pts) { free(fb); free(all_freqs); free(f_pts); return NULL; }

    for (int i = 0; i < n_freqs; i++)
        all_freqs[i] = f_max * i / (n_freqs - 1);

    double m_min = hz_to_mel_htk(0.0), m_max = hz_to_mel_htk(f_max);
    for (int i = 0; i < n_mels + 2; i++)
        f_pts[i] = mel_to_hz_htk(m_min + (m_max - m_min) * i / (n_mels + 1));

    /* Build triangular filters. */
    for (int m = 0; m < n_mels; m++) {
        double d_lo = f_pts[m + 1] - f_pts[m];
        double d_hi = f_pts[m + 2] - f_pts[m + 1];
        for (int i = 0; i < n_freqs; i++) {
            double down = (all_freqs[i] - f_pts[m]) / d_lo;
            double up = (f_pts[m + 2] - all_freqs[i]) / d_hi;
            double v = down < up ? down : up;
            fb[(size_t)m * n_freqs + i] = (float)(v > 0.0 ? v : 0.0);
        }
    }

    free(all_freqs);
    free(f_pts);
    return fb;
}


/* Twiddles are tabulated once from double-precision cos/sin. Deriving them by
 * the usual complex recurrence instead costs ~1e-3 of accuracy in the mel
 * features by the end of a 512-point transform. */
static float twiddle_re[GRANITE_N_FFT / 2];
static float twiddle_im[GRANITE_N_FFT / 2];
static int twiddle_ready = 0;

static void fft_init(int n) {
    for (int i = 0; i < n / 2; i++) {
        double ang = -2.0 * M_PI * i / n;
        twiddle_re[i] = (float)cos(ang);
        twiddle_im[i] = (float)sin(ang);
    }
    twiddle_ready = 1;
}

static void fft_radix2(float *re, float *im, int n) {
    if (!twiddle_ready) fft_init(n);

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        int half = len / 2;
        int step = n / len;              /* stride into the twiddle table */
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < half; k++) {
                float cr = twiddle_re[k * step];
                float ci = twiddle_im[k * step];
                float ur = re[i + k],        ui = im[i + k];
                float vr = re[i + k + half], vi = im[i + k + half];
                float tr = vr * cr - vi * ci;
                float ti = vr * ci + vi * cr;
                re[i + k] = ur + tr;         im[i + k] = ui + ti;
                re[i + k + half] = ur - tr;  im[i + k + half] = ui - ti;
            }
        }
    }
}


float *granite_frontend(const granite_model_t *m, const float *samples,
                        int n_samples, int *out_frames) {
    const int hop = GRANITE_HOP_LENGTH;
    const int n_fft = GRANITE_N_FFT;
    const int n_mels = GRANITE_N_MELS;
    const int n_freqs = n_fft / 2 + 1;
    const int s = GRANITE_STACK_FACTOR;

    /* Frame count: ceil to a whole stack group, matching _frontend. */
    int mel_frames = n_samples / hop;
    int n_frames = s * ((mel_frames + s - 1) / s);
    if (n_frames < s) n_frames = s;
    int need = (n_frames - 1) * hop + 1;

    /* _frontend right-pads the waveform up to `need` but never truncates it, so
     * longer input keeps its tail, which the reflect padding below then sees.
     * Trimming to `need` here would corrupt the final frames. */
    int siglen = n_samples > need ? n_samples : need;

    /* Center the STFT with reflect padding. */
    int pad = n_fft / 2;
    int padded_n = siglen + 2 * pad;
    float *x = calloc((size_t)padded_n, sizeof(float));
    if (!x) return NULL;
    for (int i = 0; i < siglen; i++)
        x[pad + i] = i < n_samples ? samples[i] : 0.0f;
    for (int i = 0; i < pad; i++) {
        int src = pad + i + 1;                        /* reflect, excluding edge */
        x[pad - 1 - i] = x[src < padded_n ? src : padded_n - 1];
        int back = pad + siglen - 2 - i;
        x[pad + siglen + i] = x[back >= 0 ? back : 0];
    }

    /* Center the periodic Hann window in the FFT buffer. */
    float window[GRANITE_N_FFT];
    memset(window, 0, sizeof(window));
    int woff = (n_fft - GRANITE_WIN_LENGTH) / 2;
    for (int i = 0; i < GRANITE_WIN_LENGTH; i++)
        window[woff + i] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * i / GRANITE_WIN_LENGTH));

    /* Store log-mel features transposed for the delta pass. */
    float *logmel = malloc((size_t)n_mels * n_frames * sizeof(float));
    float *re = malloc((size_t)n_fft * sizeof(float));
    float *im = malloc((size_t)n_fft * sizeof(float));
    float *power = malloc((size_t)n_freqs * sizeof(float));
    if (!logmel || !re || !im || !power) {
        free(x); free(logmel); free(re); free(im); free(power);
        return NULL;
    }

    for (int t = 0; t < n_frames; t++) {
        const float *src = x + (size_t)t * hop;
        for (int i = 0; i < n_fft; i++) {
            re[i] = src[i] * window[i];
            im[i] = 0.0f;
        }
        fft_radix2(re, im, n_fft);
        for (int f = 0; f < n_freqs; f++)
            power[f] = re[f] * re[f] + im[f] * im[f];

        for (int b = 0; b < n_mels; b++) {
            const float *filt = m->mel_filters + (size_t)b * n_freqs;
            double acc = 0.0;   /* Match the reference's precision. */
            for (int f = 0; f < n_freqs; f++) acc += (double)filt[f] * power[f];
            logmel[(size_t)b * n_frames + t] = (float)acc;
        }
    }
    free(x); free(re); free(im); free(power);

    /* Apply the reference's floor, clamp, and rescale. */
    float mx = -INFINITY;
    for (size_t i = 0; i < (size_t)n_mels * n_frames; i++) {
        float v = logmel[i] < 1e-10f ? 1e-10f : logmel[i];
        v = log10f(v);
        logmel[i] = v;
        if (v > mx) mx = v;
    }
    float floor_v = mx - GRANITE_LOGMEL_FLOOR_DB;
    for (size_t i = 0; i < (size_t)n_mels * n_frames; i++) {
        float v = logmel[i] < floor_v ? floor_v : logmel[i];
        logmel[i] = v / 4.0f + 1.0f;
    }

    /* Deltas along time: (-x[t-1] + x[t+1]) / 2 with replicate padding. */
    float *deltas = malloc((size_t)n_mels * n_frames * sizeof(float));
    if (!deltas) { free(logmel); return NULL; }
    for (int b = 0; b < n_mels; b++) {
        const float *row = logmel + (size_t)b * n_frames;
        float *drow = deltas + (size_t)b * n_frames;
        for (int t = 0; t < n_frames; t++) {
            float prev = row[t > 0 ? t - 1 : 0];
            float next = row[t + 1 < n_frames ? t + 1 : n_frames - 1];
            drow[t] = (next - prev) / 2.0f;
        }
    }

    /* Interleave static+delta per frame (160 dims), then stack `s` frames. */
    int out_n = n_frames / s;
    float *feats = malloc((size_t)out_n * GRANITE_INPUT_DIM * sizeof(float));
    if (!feats) { free(logmel); free(deltas); return NULL; }
    for (int o = 0; o < out_n; o++) {
        float *dst = feats + (size_t)o * GRANITE_INPUT_DIM;
        for (int k = 0; k < s; k++) {
            int t = o * s + k;
            float *slot = dst + k * (2 * n_mels);
            for (int b = 0; b < n_mels; b++) {
                slot[b] = logmel[(size_t)b * n_frames + t];
                slot[n_mels + b] = deltas[(size_t)b * n_frames + t];
            }
        }
    }

    free(logmel);
    free(deltas);
    *out_frames = out_n;
    return feats;
}
