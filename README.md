# granite.cpp

Pure C inference for [IBM Granite Speech 5.0](https://huggingface.co/ibm-granite),
a 473M-parameter CTC conformer speech-to-text model. No PyTorch, no ONNX, no
runtime dependencies beyond libc, pthreads and a BLAS.

Companion to [`qwen-asr`](../qwen-asr), which does the same for Qwen3-ASR.

## Build

```sh
make blas          # Accelerate on macOS, OpenBLAS on Linux
make mps           # Metal/MPS on Apple Silicon
```

`mps` is a superset of `blas`: it keeps BLAS for the shapes too small to be
worth a GPU dispatch, and falls back to it wholesale if no Metal device is
available. See [Backends](#backends).

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

### Streaming

`--stream` decodes incrementally, printing text as it stabilizes:

```sh
arecord -f S16_LE -r 16000 -c 1 | ./granite -d granite-speech-5.0 --stream
```

It re-encodes a sliding window and commits only frames that are settled, so
emitted text is never retracted. Output starts once ~10 s of audio has arrived
(see *Chunked decoding* below) and then follows with ~1.6 s of latency.

### Segmented

`-S <secs>` decodes long files in independent pieces, moving each cut to the
quietest point within `-W` seconds so boundaries land in pauses:

```sh
./granite -d granite-speech-5.0 -i lecture.wav -S 30
```

This avoids a single huge logits allocation (one contiguous
`frames × 16384` float buffer, ~1.4 GB per 30 minutes). It does not lower peak
RSS; the weight cache dominates that.

### Options

| Flag | Meaning |
| --- | --- |
| `-d, --model-dir DIR` | directory with `model.safetensors` + `tokenizer.json` |
| `-i, --input FILE` | 16-bit PCM WAV to transcribe |
| `--stdin` | read WAV or raw s16le 16 kHz mono from stdin |
| `-S, --segment SECS` | segmented decode (0 = whole clip, the default) |
| `-W, --cut-window SECS` | search radius for a segment cut (default 3.0) |
| `--stream` | incremental decode; implies `--stdin` |
| `--stream-chunk SECS` | audio between decodes (default 2.0) |
| `--stream-window SECS` | context re-encoded per decode (default 20.5) |
| `--stream-lookahead SECS` | audio withheld from commit (default 1.6) |
| `-t, --threads N` | worker threads (default: CPU count) |
| `--silent` | quiet stderr; the transcription still goes to stdout |
| `--debug` | verbose diagnostics, weight-cache and peak-RSS report |
| `--weight-cache MB` | cap the bf16→f32 weight cache (default 3072) |
| `--dump DIR` | write `feats.bin` / `logits.bin` for the parity harness |

## Model

The output is lowercase and unpunctuated. That is inherent to this model's
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
checkpoint's own tensor names, so it needs no `transformers`: Granite Speech 5.0
ships as `transformers 5.16.0.dev0`, which is unreleased. It is the parity
oracle, not part of the inference path.

```sh
make test        # or: ./test_granite.py
```

The harness runs both implementations over real samples and a synthetic length
sweep chosen to exercise the awkward shapes: a trailing attention block shorter
than the 128-frame context, and odd frame counts hitting the subsampling trim.

```
    feats  max|d| = 4.554e-05 over 176000 values
    logits max|d| = 1.788e-04 over 2244608 values; argmax 137/137 frames agree
  PASS jfk.wav
```

Features match to ~1e-5 and logits to ~1e-4; argmax agrees on 100% of frames in
every case, which is what determines the transcript.

## Chunked decoding

Attention is block-local over 128 encoder frames, and a frame is 80 ms, so the
model's effective receptive field is one attention block, about 10.2 s. Measured
directly: perturbing 10 ms of audio changes logits across exactly frames 0 to 127
of its block, and only weakly beyond. Everything about chunking follows from
that number.

- Segments below 10.2 s are clamped up to it.
- Streaming windows start on block boundaries, not merely frame boundaries.
  Aligning only to frames shifts the whole block grid between windows and was
  worth ~1.8% WER.
- Nothing is committed until a full block of audio exists; before that the
  window's right edge is padding rather than signal.

Cost of chunking, measured against the whole-clip decode:

| Audio | `-S 30` | `--stream` |
| --- | --- | --- |
| 11s / 3.6s samples | 0.00% | 0.00% |
| 119s real speech | 2.21% | 0.44% |
| 10 min | not measured | 0.61% |

Streaming tracks offline more closely than segmented because it re-decodes
overlapping context and commits only settled frames, where segmented decodes
disjoint pieces and concatenates. Against *ground truth* on the 119 s sample all
three modes are equivalent (offline 18.1%, stream and segment 17.2%), so
chunking costs no real accuracy.

## Performance

Apple Silicon, `make blas`, 8 threads, model file warm in the page cache:

| Audio | Inference | RTF |
| --- | --- | --- |
| 3.6s | 0.41s | 0.114 |
| 11s | 0.62s | 0.057 |
| 66s | 2.24s | 0.034 |

Roughly 9x to 29x realtime, improving with clip length as fixed costs amortize.
A 10-minute file decodes in ~20 s (RTF 0.033).

Peak RSS is ~2.4 GB by default, dominated by the bf16→f32 weight cache;
`--weight-cache 0` drops that to ~1.3 GB by reconverting weights per call.
Neither `-S` nor `--stream` reduces peak RSS meaningfully, since the weights
dominate: measured 3.0 GB whole-clip against 3.1 GB segmented on 30 minutes.

## Backends

| | `make blas` | `make mps` |
| --- | --- | --- |
| Linears | `cblas_sgemm` | `MPSMatrixMultiplication` |
| Block attention | NEON + thread pool | NEON + thread pool |
| Norms, GLU, SiLU, depthwise conv | CPU | CPU |
| Platforms | macOS, Linux | macOS on Apple Silicon |

The MPS backend allocates the weight cache and every encoder scratch buffer as
Metal shared buffers, so the GPU reads the same bytes the CPU wrote. Operands
are never uploaded per call and the 1.8 GB weight cache is not duplicated on the
device. That is the point of the design; without it the device would need its
own second copy.

It still costs some memory, measured on an M1:

| Peak RSS | 11s clip | 119s clip |
| --- | --- | --- |
| `make blas` | 2242 MB | 2506 MB |
| `make mps` | 2767 MB | 2690 MB |

The 180 to 525 MB of overhead is per-buffer page rounding across the ~390 weight
allocations, MPS internal workspaces, and the staging slots. The MPS build is
not memory-neutral.

Only the linears go to the GPU. Norms and elementwise ops stay on the CPU
because each would cost a dispatch round-trip to save very little, and attention
stays there because it was tried: a hand-written shader measured 1.00 s against
the CPU kernel's 0.27 s over a 119 s clip, so it was removed. Under `--debug`
the backend reports the device it selected:

```
$ ./granite -d granite-speech-5.0 -i audio.wav --debug
[metal] using device: Apple M2 Pro
```

If that line is absent on an `mps` build, the run went to BLAS.

## Layout

| File | Contents |
| --- | --- |
| `main.c` | CLI |
| `granite.c` | model load/free, end-to-end transcription |
| `granite_encoder.c` | conformer forward pass, weight binding |
| `granite_kernels.c` | thread pool, bf16 GEMM, norms, attention, conv |
| `granite_kernels_metal.m` | Metal backend: shared buffers, MPS GEMM |
| `granite_audio.c` | WAV I/O, FFT, mel filterbank, front-end |
| `granite_tokenizer.c` | decode-only BPE (CTC never encodes text) |
| `granite_safetensors.c` | mmap'd checkpoint reader (from `qwen-asr`) |
| `reference.py` | PyTorch parity oracle |
| `test_granite.py` | regression + parity harness |
| `tools/stream_wer.py` | WER of a chunked mode against the offline decode |

## License

Apache 2.0, matching the model.
