/*
 * granite_audio.h - WAV loading and the log-mel(+delta) front-end
 */

#ifndef GRANITE_AUDIO_H
#define GRANITE_AUDIO_H

#include <stddef.h>
#include <stdint.h>

/* Load a 16-bit PCM WAV as mono float32 in [-1, 1], resampled to 16 kHz.
 * Returns NULL on error; caller frees. */
float *granite_load_wav(const char *path, int *out_n_samples);

/* Same, from an in-memory WAV image. */
float *granite_parse_wav_buffer(const uint8_t *data, size_t size, int *out_n_samples);

/* Read audio from stdin, auto-detecting WAV vs raw s16le 16 kHz mono. */
float *granite_read_stdin(int *out_n_samples);

/* ------------------------------------------------------------ live audio --- */

/* Incremental stdin reader for --stream. A background thread appends samples
 * to a growing buffer; the decode loop waits on and copies out of it. */
typedef struct granite_live_audio granite_live_audio_t;

granite_live_audio_t *granite_live_audio_start_stdin(void);
void granite_live_audio_free(granite_live_audio_t *la);

/* Block until at least `want` samples have arrived or the stream ends.
 * Returns the total sample count now available and sets *eof. */
int granite_live_audio_wait(granite_live_audio_t *la, int want, int *eof);

/* Copy [start, start + n) out of the live buffer. */
void granite_live_audio_copy(granite_live_audio_t *la, int start, int n, float *dst);

/* Build the mel filterbank torchaudio produces for MelSpectrogram's defaults
 * (slaney-style triangular filters on an HTK mel scale, f_min=0, f_max=sr/2,
 * norm=None). Returns [n_mels, n_fft/2 + 1]; caller frees. */
float *granite_mel_filterbank(int sample_rate, int n_fft, int n_mels);

#endif /* GRANITE_AUDIO_H */
