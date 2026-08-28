/* Command-line interface for Granite Speech inference. */

#include "granite.h"
#include "granite_audio.h"
#include "granite_kernels.h"
#include "granite_tokenizer.h"

#ifdef USE_MPS
#include "granite_kernels_metal.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s -d <model_dir> [-i <audio.wav> | --stdin] [options]\n"
        "\n"
        "Options:\n"
        "  -d, --model-dir DIR   directory with model.safetensors + tokenizer.json\n"
        "  -i, --input FILE      16-bit PCM WAV to transcribe\n"
        "      --stdin           read WAV or raw s16le 16 kHz mono from stdin\n"
        "  -S, --segment SECS    segmented decode (0 = whole clip, the default).\n"
        "                        Avoids one huge logits allocation on long files;\n"
        "                        cuts are moved to the quietest point nearby\n"
        "  -W, --cut-window SECS search radius for a segment cut (default 3.0)\n"
        "      --stream          incremental decode; implies --stdin\n"
        "      --stream-chunk SECS      audio between decodes (default 2.0)\n"
        "      --stream-window SECS     context re-encoded per decode (default 20.5)\n"
        "      --stream-lookahead SECS  audio withheld from commit (default 1.6)\n"
        "  -t, --threads N       worker threads (default: physical CPUs)\n"
        "      --silent          suppress status output on stderr\n"
        "      --debug           verbose diagnostics on stderr\n"
        "      --dump DIR        write feats.bin/logits.bin for the parity harness\n"
        "      --weight-cache MB  cap on the bf16->f32 weight cache (default 3072;\n"
        "                         0 reconverts each call, trading speed for ~1.9 GB)\n"
        "  -h, --help            show this help\n",
        prog);
}

/* Run the pipeline and optionally dump raw f32 features and logits. */
static char *transcribe_and_dump(granite_model_t *m, const float *samples,
                                 int n_samples, const char *dir) {
    int n_frames = 0;
    float *feats = granite_frontend(m, samples, n_samples, &n_frames);
    if (!feats) return NULL;

    int out_frames = 0;
    float *logits = granite_encode(m, feats, n_frames, &out_frames);
    if (!logits) { free(feats); return NULL; }

    const struct { const char *name; const float *data; size_t n; } stages[] = {
        { "feats",  feats,  (size_t)n_frames * GRANITE_INPUT_DIM },
        { "logits", logits, (size_t)out_frames * GRANITE_VOCAB },
    };
    for (size_t i = 0; i < sizeof(stages) / sizeof(stages[0]); i++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s.bin", dir, stages[i].name);
        FILE *f = fopen(path, "wb");
        if (!f) {
            fprintf(stderr, "granite: cannot write %s\n", path);
            continue;
        }
        fwrite(stages[i].data, sizeof(float), stages[i].n, f);
        fclose(f);
    }
    fprintf(stderr, "granite: dumped feats[%d,%d] logits[%d,%d] to %s\n",
            n_frames, GRANITE_INPUT_DIM, out_frames, GRANITE_VOCAB, dir);

    int *ids = malloc((size_t)out_frames * sizeof(int));
    char *text = NULL;
    if (ids) {
        int n_ids = granite_ctc_greedy(logits, out_frames, GRANITE_VOCAB, ids);
        text = granite_tokenizer_decode(m->tok, ids, n_ids);
        free(ids);
    }
    free(feats);
    granite_logits_free(logits);
    return text;
}

/* Print newly committed streaming text. */
static void emit_delta(const char *delta, void *user) {
    (void)user;
    fputs(delta, stdout);
    fflush(stdout);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *input = NULL;
    const char *dump_dir = NULL;
    int use_stdin = 0, threads = 0, silent = 0, debug = 0, stream = 0;

    granite_params_t params;
    granite_default_params(&params);

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if ((!strcmp(a, "-d") || !strcmp(a, "--model-dir")) && i + 1 < argc) {
            model_dir = argv[++i];
        } else if ((!strcmp(a, "-i") || !strcmp(a, "--input")) && i + 1 < argc) {
            input = argv[++i];
        } else if (!strcmp(a, "--stdin")) {
            use_stdin = 1;
        } else if (!strcmp(a, "--stream")) {
            stream = 1;
            use_stdin = 1;
        } else if ((!strcmp(a, "-S") || !strcmp(a, "--segment")) && i + 1 < argc) {
            params.segment_sec = (float)atof(argv[++i]);
        } else if ((!strcmp(a, "-W") || !strcmp(a, "--cut-window")) && i + 1 < argc) {
            params.cut_window_sec = (float)atof(argv[++i]);
        } else if (!strcmp(a, "--stream-chunk") && i + 1 < argc) {
            params.stream_chunk_sec = (float)atof(argv[++i]);
        } else if (!strcmp(a, "--stream-window") && i + 1 < argc) {
            params.stream_window_sec = (float)atof(argv[++i]);
        } else if (!strcmp(a, "--stream-lookahead") && i + 1 < argc) {
            params.stream_lookahead_sec = (float)atof(argv[++i]);
        } else if ((!strcmp(a, "-t") || !strcmp(a, "--threads")) && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (!strcmp(a, "--silent")) {
            silent = 1;
        } else if (!strcmp(a, "--debug")) {
            debug = 1;
        } else if (!strcmp(a, "--dump") && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (!strcmp(a, "--weight-cache") && i + 1 < argc) {
            granite_set_weight_cache_limit((size_t)atoll(argv[++i]) << 20);
        } else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "granite: unknown argument '%s'\n", a);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_dir || (!input && !use_stdin)) {
        usage(argv[0]);
        return 1;
    }

    granite_set_threads(threads > 0 ? threads : granite_get_num_cpus());

#ifdef USE_MPS
    /* Configure diagnostics before loading the model. */
    granite_metal_set_verbose(silent ? 0 : (debug ? 2 : 1));
#endif

    double t0 = now_sec();
    granite_model_t *m = granite_load(model_dir, debug);
    if (!m) return 1;
    double t_load = now_sec() - t0;

    /* Streaming decodes incrementally and prints committed text. */
    if (stream) {
        granite_live_audio_t *la = granite_live_audio_start_stdin();
        if (!la) {
            granite_free(m);
            return 1;
        }
        double t1 = now_sec();
        char *text = granite_transcribe_stream(m, la, &params, emit_delta, NULL);
        double t_infer = now_sec() - t1;
        printf("\n");
        fflush(stdout);
        if (!silent)
            fprintf(stderr, "stream | infer %.2fs\n", t_infer);
        free(text);
        granite_live_audio_free(la);
        granite_free(m);
        granite_kernels_shutdown();
        return 0;
    }

    int n_samples = 0;
    float *samples = use_stdin ? granite_read_stdin(&n_samples)
                               : granite_load_wav(input, &n_samples);
    if (!samples) {
        granite_free(m);
        return 1;
    }

    double t1 = now_sec();
    char *text;
    if (dump_dir)
        text = transcribe_and_dump(m, samples, n_samples, dump_dir);
    else if (params.segment_sec > 0.0f)
        text = granite_transcribe_segmented(m, samples, n_samples, &params, NULL, NULL);
    else
        text = granite_transcribe(m, samples, n_samples);
    double t_infer = now_sec() - t1;

    if (!text) {
        fprintf(stderr, "granite: transcription failed\n");
        free(samples);
        granite_free(m);
        return 1;
    }

    /* --silent suppresses status output, not the transcription. */
    printf("%s\n", text);

    if (!silent) {
        double audio_sec = (double)n_samples / GRANITE_SAMPLE_RATE;
        fprintf(stderr, "load %.2fs | audio %.2fs | infer %.2fs | RTF %.3f\n",
                t_load, audio_sec, t_infer,
                audio_sec > 0 ? t_infer / audio_sec : 0.0);
    }
    if (debug) {
        struct rusage ru;
        getrusage(RUSAGE_SELF, &ru);
        /* ru_maxrss is bytes on macOS, kilobytes on Linux. */
#ifdef __APPLE__
        double rss_mb = ru.ru_maxrss / (1024.0 * 1024.0);
#else
        double rss_mb = ru.ru_maxrss / 1024.0;
#endif
        fprintf(stderr, "granite: weight cache %.0f MB | peak RSS %.0f MB\n",
                granite_weight_cache_bytes() / (1024.0 * 1024.0), rss_mb);
    }

    free(text);
    free(samples);
    granite_free(m);
    granite_kernels_shutdown();
    return 0;
}
