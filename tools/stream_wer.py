#!/usr/bin/env python3
"""Word error rate of a mode (segmented / streaming) against offline decode.

Offline is the reference here, not ground truth: the question this answers is
"how much does chunking cost?", not "how good is the model?".

Usage: tools/stream_wer.py <wav> [--mode stream|segment] [extra granite args...]
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(HERE, "granite")
MODEL = os.path.join(HERE, "granite-speech-5.0")


def run(args, stdin_data=None):
    p = subprocess.run([BINARY, "-d", MODEL, "--silent"] + args,
                       input=stdin_data, capture_output=True)
    if p.returncode != 0:
        sys.exit(f"granite failed: {p.stderr.decode()}")
    return p.stdout.decode().strip()


def wer(ref, hyp):
    """Levenshtein over words; returns (errors, ref_len, ops)."""
    r, h = ref.split(), hyp.split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    # Walk back to list the actual differences.
    ops, i, j = [], len(r), len(h)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            if r[i - 1] != h[j - 1]:
                ops.append(f"sub {r[i-1]!r}->{h[j-1]!r}")
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(f"del {r[i-1]!r}")
            i -= 1
        else:
            ops.append(f"ins {h[j-1]!r}")
            j -= 1
    return d[len(r)][len(h)], len(r), list(reversed(ops))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--mode", choices=["stream", "segment"], default="stream")
    args, extra = ap.parse_known_args()
    args.extra = extra

    ref = run(["-i", args.wav])
    if args.mode == "stream":
        with open(args.wav, "rb") as f:
            hyp = run(["--stream"] + args.extra, stdin_data=f.read())
    else:
        hyp = run(["-i", args.wav] + args.extra)

    errs, n, ops = wer(ref, hyp)
    print(f"{args.mode} {' '.join(args.extra):24s} "
          f"WER vs offline: {errs}/{n} = {100.0 * errs / max(n, 1):.2f}%")
    for op in ops[:10]:
        print(f"    {op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
