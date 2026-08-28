# granite.cpp

`granite.cpp` is a lightweight, pure C inference engine for [IBM Granite Speech 5.0](https://huggingface.co/ibm-granite), a 473M-parameter CTC conformer speech-to-text model.

It runs locally with minimal dependencies (standard C library, POSIX threads, and a BLAS library), with no Python runtime, PyTorch, or ONNX needed for inference.

### Key Characteristics
- **Architecture**: Encoder-only Conformer with 16 blocks (blocks 0 and 1 subsample 2x each in time, yielding 80 ms per encoder frame).
- **Decoding**: Greedy CTC collapse (no autoregressive decoder, no KV cache, no sampling).
- **Output format**: Always lowercase and unpunctuated, reflecting the model's 16,384-token vocabulary.
- **Hardware backends**: CPU (NEON / generic C with BLAS) and an optional Apple Silicon Metal/MPS backend.

---

## Quick Start & Usage

### 1. Download Model Files

You need `model.safetensors` and `tokenizer.json` from the IBM Granite Speech 5.0 repository:

```sh
# Example using huggingface-cli:
huggingface-cli download ibm-granite/granite-speech-5.0 \
  model.safetensors tokenizer.json \
  --local-dir ./granite-speech-5.0
```

### 2. Transcribing Audio Files

Audio input must be 16-bit PCM (WAV format, or raw mono via standard input). 16 kHz mono is recommended.

```sh
# Transcribe a single file (whole-clip offline mode)
./granite -d granite-speech-5.0 -i samples/jfk.wav
```

Output:
```text
and so my fellow americans ask not what your country can do for you ask what you can do for your country
load 0.01s | audio 11.00s | infer 0.62s | RTF 0.057
```

You can also pipe audio via stdin:
```sh
cat samples/jfk.wav | ./granite -d granite-speech-5.0 --stdin
```

### 3. Real-Time Streaming & Segmented Modes

#### Streaming Mode (`--stream`)
Decodes audio incrementally from stdin (e.g. microphone capture), emitting stable text without waiting for the full recording to finish:

```sh
# Linux (ALSA)
arecord -f S16_LE -r 16000 -c 1 | ./granite -d granite-speech-5.0 --stream

# macOS (sox)
rec -t raw -r 16000 -c 1 -b 16 -e signed-integer - | ./granite -d granite-speech-5.0 --stream
```

#### Segmented Processing (`-S <seconds>`)
For long audio recordings (lectures, podcasts), segmented decoding processes the file in chunks while automatically snapping cut boundaries to quiet pauses:

```sh
./granite -d granite-speech-5.0 -i lecture.wav -S 30
```

### 4. CLI Options Reference

| Option | Description | Default |
| --- | --- | --- |
| `-d, --model-dir DIR` | Path to directory containing `model.safetensors` and `tokenizer.json` | `.` |
| `-i, --input FILE` | Path to input 16-bit PCM WAV file | — |
| `--stdin` | Read WAV or raw s16le (16 kHz mono) from stdin | — |
| `-S, --segment SECS` | Process audio in chunks of ~N seconds (`0` = entire file at once) | `0` |
| `-W, --cut-window SECS` | Search window around chunk boundary to find the quietest pause point | `3.0` |
| `--stream` | Incremental low-latency stream decoding (implies `--stdin`) | disabled |
| `--stream-chunk SECS` | Audio step interval between partial decodes | `2.0` |
| `--stream-window SECS` | Re-encoded sliding window context | `20.5` |
| `--stream-lookahead SECS` | Audio withheld from commit to ensure stability | `1.6` |
| `-t, --threads N` | Number of worker threads | CPU core count |
| `--silent` | Silence progress and runtime stats on stderr (stdout transcription remains) | disabled |
| `--debug` | Print verbose timing, memory usage, and weight cache statistics | disabled |
| `--weight-cache MB` | Maximum RAM allocated for bf16→f32 converted weights (`0` to disable caching) | `3072` |
| `--dump DIR` | Dump raw intermediate tensors (`feats.bin`, `logits.bin`) for parity testing | disabled |

---

## Compilation

### Prerequisites
- C11 compiler (`clang` or `gcc`)
- `make`
- A BLAS library:
  - **macOS**: uses Apple Accelerate framework (built-in).
  - **Linux**: requires OpenBLAS (`libopenblas-dev` on Debian/Ubuntu, `openblas-devel` on Fedora/RHEL).

### Build Targets

```sh
# CPU build (Accelerate on macOS, OpenBLAS on Linux)
make blas

# Apple Silicon GPU/MPS build (Metal Matrix Multiplication + CPU norms/attention)
make mps

# Run parity and test suite (requires Python 3 venv with torch/torchaudio/tokenizers)
make test

# Clean artifacts
make clean
```

---

## Streaming and Chunked Processing

### The 10.24-Second Receptive Field

Granite Speech 5.0 computes block-local self-attention over **128 encoder frames**. Because subsampling reduces the 10 ms audio frame rate to 80 ms per encoder frame, each attention block spans exactly:

$$\text{Block size} = 128 \times 80\text{ ms} = 10.24\text{ seconds}$$

A change in any audio sample only influences the logits within its 10.24 s attention block. Both streaming and segmented modes are built directly around this property.

### How Modes Work

```
1. Offline (Whole Clip)
   [==================== Full Audio ====================] -> Single Forward Pass -> CTC Greedy Collapse

2. Segmented (-S 30)
   [--- ~30s Block ---] cut at pause [--- ~30s Block ---] cut at pause [--- ~30s Block ---]
       (Each segment decoded independently; prevents huge contiguous logits buffers on long audio)

3. Streaming (--stream)
   Sliding window re-encodes context across block boundaries:
   Window 0: [====== Context ======|...lookahead...]
                                   └── Commit settled tokens ──> emit to stdout
   Window 1:         [====== Context ======|...lookahead...]
                                           └── Commit settled tokens ──> emit to stdout
```

### Technical Invariants in Streaming

- **Block-aligned sliding windows**: Streaming windows always start on an attention block boundary (multiples of 10.24 s). Aligning windows to arbitrary frame boundaries shifts the attention grid and increases word error rate (WER).
- **Lookahead buffer**: The trailing `1.6 s` (`--stream-lookahead`) of the active window is held uncommitted until the next step arrives, avoiding boundary distortions caused by edge padding.
- **Stable prefix commits**: Tokens are committed incrementally once they move past the lookahead horizon. Already-committed frames are never re-evaluated or overwritten, ensuring text is emitted monotonically with zero flickering or rollbacks.
- **Segmented cutting**: In `-S` mode, chunk splits are placed at local energy minima within `-W` seconds to avoid severing words mid-phoneme.

---

## Project Structure

```
granite.cpp/
├── main.c                  # CLI entrypoint and argument parsing
├── granite.c               # High-level pipeline and orchestration
├── granite.h               # Public C API and model structures
├── granite_encoder.c       # Conformer encoder forward pass & layer logic
├── granite_kernels.c       # Thread pool, GEMM bindings, NEON attention, norms, conv
├── granite_kernels_metal.m # Metal/MPS shared-buffer GEMM backend
├── granite_audio.c         # WAV parsing, STFT, Hann window, mel filterbanks
├── granite_tokenizer.c     # Decode-only BPE tokenizer
├── granite_safetensors.c   # Fast mmap-based safetensors loader
├── reference.py            # PyTorch reference implementation for parity checks
└── test_granite.py         # Test suite comparing C output vs reference implementation
```

## License

MIT

## Use of LLMs

The project is largely vibecoded with Opus 5.0 at low thinking effort,
running under human supervision, starting from my fork of [qwen-asr](github.com/mseri/qwen-asr)
and proceeding from there. Some manual cleanup started happening only
after everything was working fine, so don't be surprise about leftover
or past AI slop in the text.
