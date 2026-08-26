#!/usr/bin/env python3
"""Regression harness: compare ./granite against reference.py.

For each sample it checks three things, in increasing strictness:
  1. the transcribed text matches
  2. the front-end features match (isolates mel/delta/stack numerics)
  3. the CTC logits match, and their argmax agrees frame-for-frame

The logits are computed in f32 in C and in f32 in torch but from bf16 weights,
so small differences are expected; argmax agreement is the decision-relevant
check and must be exact.

Usage: ./test_granite.py [--binary ./granite] [--model-dir granite-speech-5.0]
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(HERE, ".venv", "bin", "python")

SAMPLES = [
    ("../qwen-asr/samples/jfk.wav",
     "and so my fellow americans ask not what your country can do for you "
     "ask what you can do for your country"),
    ("../qwen-asr/samples/test_speech.wav",
     "hello this is a test of the voxtral speech to text system"),
]

INPUT_DIM = 320
VOCAB = 16384


def read_bin(path):
    with open(path, "rb") as f:
        raw = f.read()
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def compare(name, a, b, tol):
    """Return (ok, max_abs_diff, detail)."""
    if len(a) != len(b):
        return False, float("inf"), f"length {len(a)} vs {len(b)}"
    worst = 0.0
    for x, y in zip(a, b):
        d = abs(x - y)
        if d > worst:
            worst = d
    return worst <= tol, worst, ""


def argmax_rows(flat, cols):
    out = []
    for i in range(0, len(flat), cols):
        row = flat[i:i + cols]
        best, best_v = 0, row[0]
        for j in range(1, cols):
            if row[j] > best_v:
                best_v, best = row[j], j
        out.append(best)
    return out


def make_synthetic(path, n_samples, seed=1234):
    """Deterministic pseudo-random speech-ish noise, for shape edge cases."""
    state = seed
    frames = bytearray()
    for _ in range(n_samples):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        frames += struct.pack("<h", (state % 20000) - 10000)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(bytes(frames))


def run_case(binary, model_dir, wav, expected, tmp, label):
    c_dir = os.path.join(tmp, "c")
    py_dir = os.path.join(tmp, "py")
    os.makedirs(c_dir, exist_ok=True)
    os.makedirs(py_dir, exist_ok=True)

    c = subprocess.run([binary, "-d", model_dir, "-i", wav, "--dump", c_dir,
                        "--silent"], capture_output=True, text=True)
    if c.returncode != 0:
        print(f"  FAIL {label}: granite exited {c.returncode}\n{c.stderr}")
        return False
    c_text = c.stdout.strip()

    p = subprocess.run([PYTHON, os.path.join(HERE, "reference.py"), wav,
                        "--dump", py_dir], capture_output=True, text=True)
    if p.returncode != 0:
        print(f"  FAIL {label}: reference.py exited {p.returncode}\n{p.stderr}")
        return False
    # reference.py prints exactly one stdout line, the transcription, which is
    # empty for non-speech input.
    py_text = p.stdout.strip()

    ok = True

    if expected is not None and c_text != expected:
        print(f"  FAIL {label}: text mismatch\n    got      {c_text!r}\n"
              f"    expected {expected!r}")
        ok = False
    if c_text != py_text:
        print(f"  FAIL {label}: C vs torch text\n    C     {c_text!r}\n"
              f"    torch {py_text!r}")
        ok = False

    c_feats = read_bin(os.path.join(c_dir, "feats.bin"))
    py_feats = read_bin(os.path.join(py_dir, "feats.bin"))
    good, worst, detail = compare("feats", c_feats, py_feats, 2e-3)
    print(f"    feats  max|d| = {worst:.3e} over {len(c_feats)} values {detail}")
    if not good:
        print(f"  FAIL {label}: front-end mismatch {detail}")
        ok = False

    c_log = read_bin(os.path.join(c_dir, "logits.bin"))
    py_log = read_bin(os.path.join(py_dir, "logits.bin"))
    if len(c_log) != len(py_log):
        print(f"  FAIL {label}: logits length {len(c_log)} vs {len(py_log)}")
        return False
    _, worst, _ = compare("logits", c_log, py_log, 0.0)
    c_arg = argmax_rows(c_log, VOCAB)
    py_arg = argmax_rows(py_log, VOCAB)
    agree = sum(1 for x, y in zip(c_arg, py_arg) if x == y)
    print(f"    logits max|d| = {worst:.3e} over {len(c_log)} values; "
          f"argmax {agree}/{len(c_arg)} frames agree")
    if agree != len(c_arg):
        print(f"  FAIL {label}: argmax disagrees on "
              f"{len(c_arg) - agree} frame(s)")
        ok = False

    if ok:
        print(f"  PASS {label}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.path.join(HERE, "granite"))
    ap.add_argument("--model-dir", default=os.path.join(HERE, "granite-speech-5.0"))
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        print(f"missing {args.binary}; run `make blas` first")
        return 1
    if not os.path.exists(PYTHON):
        print(f"missing {PYTHON}; create the reference venv first")
        return 1

    all_ok = True
    with tempfile.TemporaryDirectory() as tmp:
        print("Real samples:")
        for rel, expected in SAMPLES:
            wav = os.path.join(HERE, rel)
            if not os.path.exists(wav):
                print(f"  SKIP {rel} (not found)")
                continue
            all_ok &= run_case(args.binary, args.model_dir, wav, expected, tmp,
                               os.path.basename(rel))

        # Shape edge cases. Frame counts drive the two paths most likely to
        # diverge: the true-length trailing attention block (T % 128 != 0) and
        # the odd-length trim in the subsampling blocks.
        print("\nSynthetic length sweep:")
        for n_samples, label in [
            (16000 * 1, "1.0s"),
            (16000 * 4 + 137, "4.0s+odd"),
            (16000 * 21, "21.0s"),
            (160 * 512 * 2, "exact-block"),
            (16000 // 2, "0.5s-short"),
        ]:
            wav = os.path.join(tmp, f"syn_{label}.wav")
            make_synthetic(wav, n_samples)
            all_ok &= run_case(args.binary, args.model_dir, wav, None, tmp, label)

    print("\nAll checks passed." if all_ok else "\nFAILURES above.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
