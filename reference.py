"""Self-contained torch reference for Granite Speech 5.0 CTC conformer.

Built directly from the checkpoint's own tensor names, so it needs no
`transformers` (the checkpoint is transformers 5.16.0.dev0, unreleased).
Semantics come from:
  - transformers granite_speech modeling (FeedForward / Attention / ConvModule /
    Block / CTCEncoder)
  - granite-speech-5.0/granite_encoder.py  (Linear pointwise convs, separate
    QKV, true-length trailing attention block, block-0/1 subsampling)
  - granite-speech-5.0/processing_ctc_conformer.py (log-mel + delta front-end)

This is the parity oracle for the C port: --dump writes intermediate tensors.
"""

import argparse
import json
import math
import os
import struct
import sys

import torch
import torchaudio
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "granite-speech-5.0")

# --- config (granite-speech-5.0/config.json + configuration_ctc_conformer.py) ---
INPUT_DIM = 320
NUM_LAYERS = 16
HIDDEN = 1024
FF_MULT = 4
NUM_HEADS = 8
DIM_HEAD = 128
OUTPUT_DIM = 16384
CONTEXT_SIZE = 128
MAX_POS_EMB = 512
CONV_KERNEL = 7
CONV_EXPANSION = 2
SUBSAMPLE_LAYERS = {0, 1}

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 80
STACK_FACTOR = 2
DELTA_WIN_LENGTH = 3
LOGMEL_FLOOR_DB = 8.0


# --------------------------------------------------------------- weights ---
def load_safetensors(path):
    """Minimal safetensors reader -> dict[str, torch.Tensor] (f32)."""
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hdr_len))
        header.pop("__metadata__", None)
        blob = f.read()
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16,
              "I64": torch.int64}
    out = {}
    for name, meta in header.items():
        s, e = meta["data_offsets"]
        t = torch.frombuffer(bytearray(blob[s:e]), dtype=dtypes[meta["dtype"]])
        out[name] = t.reshape(meta["shape"]).float()
    return out


# -------------------------------------------------------------- frontend ---
class FrontEnd(nn.Module):
    """log-mel(+delta) + frame stacking -> [B, T', 320]."""

    def __init__(self):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH, n_mels=N_MELS,
        )

    def forward(self, x):
        s = STACK_FACTOR
        mel_frames = x.shape[1] // HOP_LENGTH
        n_frames = s * -(-mel_frames // s)
        need = (n_frames - 1) * HOP_LENGTH + 1
        if x.shape[1] < need:
            x = nn.functional.pad(x, (0, need - x.shape[1]))
        mel = self.melspec(x)[..., :n_frames]
        logmel = mel.clip(min=1e-10).log10()
        mx = logmel.amax(dim=(-2, -1), keepdim=True)
        logmel = torch.maximum(logmel, mx - LOGMEL_FLOOR_DB).div(4).add(1)
        deltas = torchaudio.functional.compute_deltas(logmel, win_length=DELTA_WIN_LENGTH)
        logmel = torch.cat((logmel, deltas), dim=-2)
        logmel = logmel.transpose(-2, -1).contiguous()
        return logmel.reshape(logmel.shape[0], -1, s * logmel.shape[-1])


# --------------------------------------------------------------- encoder ---
def feed_forward(x, w, p):
    """pre_norm -> linear1 -> silu -> linear2. `p` is the checkpoint prefix."""
    h = nn.functional.layer_norm(x, (HIDDEN,), w[f"{p}norm"], w[f"{p}norm_b"])
    h = nn.functional.linear(h, w[f"{p}l1"], w[f"{p}l1_b"])
    h = nn.functional.silu(h)
    return nn.functional.linear(h, w[f"{p}l2"], w[f"{p}l2_b"])


def attention(x, w, dists, rel_emb):
    """Block-local attention with Shaw relative position bias.

    Full blocks of CONTEXT_SIZE, plus a trailing block computed at its TRUE
    length nr (granite_encoder._SeparateQKVAttention), not right-padded.
    """
    h = nn.functional.layer_norm(x, (HIDDEN,), w["norm_att"], w["norm_att_b"])
    n = h.shape[1]
    c = CONTEXT_SIZE
    nb_full, nr = n // c, n % c
    scale = DIM_HEAD ** -0.5

    q = nn.functional.linear(h, w["q"])
    k = nn.functional.linear(h, w["k"])
    v = nn.functional.linear(h, w["v"])

    def block_attn(qb, kb, vb, blk):
        b = qb.shape[0]
        g = b * (qb.shape[1] // blk)
        qf, kf, vf = (t.reshape(g, blk, NUM_HEADS, DIM_HEAD).transpose(1, 2)
                      for t in (qb, kb, vb))
        rel = rel_emb[dists[:blk, :blk]]                       # [blk, blk, d]
        bias = torch.einsum("ghcd,crd->ghcr", qf, rel) * scale
        out = nn.functional.scaled_dot_product_attention(qf, kf, vf, attn_mask=bias,
                                                         scale=scale)
        return out.transpose(1, 2).reshape(b, -1, NUM_HEADS * DIM_HEAD)

    outs = []
    L = nb_full * c
    if nb_full:
        outs.append(block_attn(q[:, :L], k[:, :L], v[:, :L], c))
    if nr:
        outs.append(block_attn(q[:, L:], k[:, L:], v[:, L:], nr))
    out = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
    return nn.functional.linear(out[:, :n, :], w["o"], w["o_b"])


def conv_module(x, w, stride):
    """LayerNorm -> pointwise_lin1 -> GLU -> depthwise conv -> BN -> SiLU -> lin2."""
    inner = HIDDEN * CONV_EXPANSION
    h = nn.functional.layer_norm(x, (HIDDEN,), w["norm_conv"], w["norm_conv_b"])
    h = nn.functional.glu(nn.functional.linear(h, w["pw1"], w["pw1_b"]), dim=-1)
    h = h.transpose(1, 2)
    pad = CONV_KERNEL // 2
    # Stock GraniteSpeechConformerDepthWiseConv1d pads (pad, pad - (k+1) % 2);
    # for the odd k=7 used here that is a symmetric (3, 3).
    h = nn.functional.pad(h, (pad, pad - (CONV_KERNEL + 1) % 2))
    h = nn.functional.conv1d(h, w["dw"], None, stride=stride, groups=inner)
    h = nn.functional.batch_norm(h, w["bn_mean"], w["bn_var"], w["bn_w"], w["bn_b"],
                                 training=False, eps=1e-5)
    h = nn.functional.silu(h).transpose(1, 2)
    return nn.functional.linear(h, w["pw2"], w["pw2_b"])


def encoder_forward(feats, W, dump=None):
    """[B, T', 320] -> CTC logits [B, T'', 16384]."""
    seq = torch.arange(CONTEXT_SIZE)
    dists = torch.clamp(seq.view(-1, 1) - seq.view(1, -1),
                        -CONTEXT_SIZE, CONTEXT_SIZE) + MAX_POS_EMB

    x = nn.functional.linear(feats, W["encoder.input_linear.weight"],
                             W["encoder.input_linear.bias"])
    if dump is not None:
        dump["input_linear"] = x.clone()

    for i in range(NUM_LAYERS):
        p = f"encoder.layers.{i}."
        lw = {
            "norm_att": W[p + "norm_self_att.weight"], "norm_att_b": W[p + "norm_self_att.bias"],
            "q": W[p + "self_attn.q_proj.weight"], "k": W[p + "self_attn.k_proj.weight"],
            "v": W[p + "self_attn.v_proj.weight"],
            "o": W[p + "self_attn.o_proj.weight"], "o_b": W[p + "self_attn.o_proj.bias"],
            "norm_conv": W[p + "norm_conv.weight"], "norm_conv_b": W[p + "norm_conv.bias"],
            "pw1": W[p + "conv.pointwise_lin1.weight"], "pw1_b": W[p + "conv.pointwise_lin1.bias"],
            "pw2": W[p + "conv.pointwise_lin2.weight"], "pw2_b": W[p + "conv.pointwise_lin2.bias"],
            "dw": W[p + "conv.depthwise_conv.weight"],
            "bn_w": W[p + "conv.norm.weight"], "bn_b": W[p + "conv.norm.bias"],
            "bn_mean": W[p + "conv.norm.running_mean"], "bn_var": W[p + "conv.norm.running_var"],
        }
        for tag, ck in (("ff1", "feed_forward1"), ("ff2", "feed_forward2")):
            lw[f"{tag}norm"] = W[p + f"norm_{ck}.weight"]
            lw[f"{tag}norm_b"] = W[p + f"norm_{ck}.bias"]
            lw[f"{tag}l1"] = W[p + f"{ck}.linear1.weight"]
            lw[f"{tag}l1_b"] = W[p + f"{ck}.linear1.bias"]
            lw[f"{tag}l2"] = W[p + f"{ck}.linear2.weight"]
            lw[f"{tag}l2_b"] = W[p + f"{ck}.linear2.bias"]
        rel_emb = W[p + "self_attn.rel_pos_emb.weight"]

        subsample = i in SUBSAMPLE_LAYERS
        x = 0.5 * feed_forward(x, lw, "ff1") + x
        x = attention(x, lw, dists, rel_emb) + x
        conv_out = conv_module(x, lw, stride=2 if subsample else 1)
        if subsample:
            t_half = x.shape[1] // 2
            x = x[:, : 2 * t_half].reshape(x.shape[0], t_half, 2, x.shape[2]).mean(dim=2)
            x = conv_out[:, :t_half] + x
        else:
            x = conv_out + x
        x = 0.5 * feed_forward(x, lw, "ff2") + x
        x = nn.functional.layer_norm(x, (HIDDEN,), W[p + "norm_out.weight"],
                                     W[p + "norm_out.bias"])
        if dump is not None:
            dump[f"layer{i}"] = x.clone()

        if i + 1 == NUM_LAYERS // 2:
            mid = nn.functional.linear(x, W["encoder.out.weight"], W["encoder.out.bias"])
            x = x + nn.functional.linear(mid.softmax(dim=-1), W["encoder.out_mid.weight"],
                                         W["encoder.out_mid.bias"])

    return nn.functional.linear(x, W["encoder.out.weight"], W["encoder.out.bias"])


def ctc_greedy(logits):
    """argmax -> collapse repeats -> drop blank (id 0)."""
    ids = logits.argmax(dim=-1)[0].tolist()
    out, prev = [], -1
    for i in ids:
        if i != prev and i != 0:
            out.append(i)
        prev = i
    return out


# ------------------------------------------------------------------ main ---
def read_wav(path):
    """16-bit PCM WAV -> mono [1, T] float32 in [-1, 1], resampled to 16 kHz."""
    import wave

    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2, "expected 16-bit PCM"
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    pcm = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
    wav = pcm.reshape(-1, ch).mean(dim=1)[None, :]
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--dump", help="directory to write intermediate tensors as .bin")
    args = ap.parse_args()

    W = load_safetensors(os.path.join(MODEL_DIR, "model.safetensors"))
    audio = read_wav(args.wav)
    feats = FrontEnd()(audio)

    dump = {} if args.dump else None
    with torch.no_grad():
        logits = encoder_forward(feats, W, dump=dump)
    ids = ctc_greedy(logits)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    print(tok.decode(ids).strip())

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        dump["feats"] = feats
        dump["logits"] = logits
        for name, t in dump.items():
            t.contiguous().numpy().astype("float32").tofile(
                os.path.join(args.dump, name + ".bin"))
            # stderr, so stdout stays exactly one line: the transcription
            # (which is empty for non-speech input).
            print(f"  dumped {name} {tuple(t.shape)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
