# granite.cpp

Pure C inference for [IBM Granite Speech 5.0](https://huggingface.co/ibm-granite),
a 473M-parameter CTC conformer speech-to-text model. No PyTorch, no ONNX, no
runtime dependencies beyond libc, pthreads and a BLAS.

Companion to [`qwen-asr`](../qwen-asr), which does the same for Qwen3-ASR.

## Build

```sh
make blas          # Accelerate on macOS, OpenBLAS on Linux
```

## Run

```sh
./granite -d granite-speech-5.0 -i audio.wav
```

```
$ ./granite -d granite-speech-5.0 -i samples/jfk.wav
and so my fellow americans ask not what your country can do for you ask what you can do for your country
load 0.01s | audio 11.00s | infer 0.62s | RTF 0.057
```

Audio can also come from stdin, as either a WAV stream or raw s16le 16 kHz mono:

```sh
cat audio.wav | ./granite -d granite-speech-5.0 --stdin
```

### Options

| Flag | Meaning |
| --- | --- |
| `-d, --model-dir DIR` | directory with `model.safetensors` + `tokenizer.json` |
| `-i, --input FILE` | 16-bit PCM WAV to transcribe |
| `--stdin` | read WAV or raw s16le 16 kHz mono from stdin |
| `-t, --threads N` | worker threads (default: CPU count) |
| `--silent` | quiet stderr; the transcription still goes to stdout |
| `--debug` | verbose diagnostics, weight-cache and peak-RSS report |
| `--weight-cache MB` | cap the bf16→f32 weight cache (default 3072) |
| `--dump DIR` | write `feats.bin` / `logits.bin` for the parity harness |

## Model

The output is **lowercase and unpunctuated** — that is inherent to this model's
16k-entry CTC vocabulary, not a limitation of this implementation.

| | |
| --- | --- |
| Parameters | 473M, bf16 (946 MB) |
| Front-end | 16 kHz → STFT(512/400/160) → 80 mel → +deltas → stack 2 → 320 dims |
| Encoder | `input_linear` 320→1024, 16 conformer blocks, CTC head 1024→16384 |
| Subsampling | blocks 0 and 1 halve time (4× total) → 80 ms per encoder frame |
| Attention | block-local, context 128, Shaw relative-position bias |
| Decoding | greedy CTC, blank = id 0 |

Unlike Qwen3-ASR there is no autoregressive decoder, KV cache, or sampling: one
encoder pass produces all logits, and the transcript is a greedy collapse of
their argmax.

Each conformer block is
`x += ½·FF1(x)` → `x += Attn(x)` → `x = Conv(x) + x` → `x += ½·FF2(x)` → `LayerNorm`,
where the conv module uses Linear pointwise projections with GLU, a depthwise
k=7 convolution, and BatchNorm. After block 8 the CTC head output is softmaxed
and folded back in through `out_mid`.

## Correctness

`reference.py` is a self-contained PyTorch implementation built from the
checkpoint's own tensor names — it needs no `transformers`, since Granite Speech
5.0 ships as `transformers 5.16.0.dev0`, which is unreleased. It is the parity
oracle, not part of the inference path.

```sh
make test        # or: ./test_granite.py
```

The harness runs both implementations over real samples and a synthetic length
sweep chosen to exercise the awkward shapes — a trailing attention block shorter
than the 128-frame context, and odd frame counts hitting the subsampling trim:

```
    feats  max|d| = 4.554e-05 over 176000 values
    logits max|d| = 1.788e-04 over 2244608 values; argmax 137/137 frames agree
  PASS jfk.wav
```

Features match to ~1e-5 and logits to ~1e-4; argmax agrees on **100% of frames**
in every case, which is what actually determines the transcript.

## Performance

Apple Silicon, `make blas`, 8 threads, model file warm in the page cache:

| Audio | Inference | RTF |
| --- | --- | --- |
| 3.6s | 0.41s | 0.114 |
| 11s | 0.62s | 0.057 |
| 66s | 2.24s | 0.034 |

Roughly 9–29× realtime, improving with clip length as fixed costs amortize.
Peak RSS is ~2.4 GB by default; `--weight-cache 0` drops that to ~1.3 GB by
reconverting weights per call instead of holding them as f32.

## Layout

| File | Contents |
| --- | --- |
| `main.c` | CLI |
| `granite.c` | model load/free, end-to-end transcription |
| `granite_encoder.c` | conformer forward pass, weight binding |
| `granite_kernels.c` | thread pool, bf16 GEMM, norms, attention, conv |
| `granite_audio.c` | WAV I/O, FFT, mel filterbank, front-end |
| `granite_tokenizer.c` | decode-only BPE (CTC never encodes text) |
| `granite_safetensors.c` | mmap'd checkpoint reader (from `qwen-asr`) |
| `reference.py` | PyTorch parity oracle |
| `test_granite.py` | regression + parity harness |

## License

Apache 2.0, matching the model.
