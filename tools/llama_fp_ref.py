#!/usr/bin/env python3
"""FP numpy reference forward for Llama 3.2 1B — loads the GGUF directly,
dequantizes weights to fp32 (NO ternarization), NORM RoPE theta=500000.

Purpose: confirm the forward ALGORITHM at full precision reproduces the
llama-cli golden tokens. If it does, the algorithm is correct and any
divergence in the C++ ter forward is a C++ bug to diff against this.

Also prints pos=0 top-5 logits + the first hidden vector stats so the C++
forward can be diffed operation-by-operation.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import gguf
import math

GGUF = (sys.argv[1] if len(sys.argv) > 1
        else "/Users/pc/osito-a-models/downloads/llama-3.2-1b-instruct/"
             "llama-3.2-1b-instruct-q8_0.gguf")
N_GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 16
GOLDEN = [28783, 25, 220, 679, 24, 11, 220, 2366, 15, 11, 220, 2366, 16, 11, 220, 2366]

r = gguf.GGUFReader(GGUF)
tensors = {t.name: t for t in r.tensors}

def deq(name):
    t = tensors[name]
    return gguf.dequantize(t.data, t.tensor_type).astype(np.float32), t.shape

# Config (Llama 3.2 1B).
H, F, Nl, Nh, Nkv, Hd, V = 2048, 8192, 16, 32, 8, 64, 128256
THETA, EPS = 500000.0, 1e-5

tok_embd, te_shape = deq("token_embd.weight")  # ggml shape [H, V] -> data [V, H]
tok_embd = tok_embd.reshape(V, H)
output_norm, _ = deq("output_norm.weight")

def load_w(name):
    # ggml stores [out, in] as data shape (out, in) row-major (gguf.dequantize
    # returns it already as [out][in]). Returns [out, in] fp32.
    w, shp = deq(name)
    # shp is ggml [ne0=in, ne1=out]; dequantized data is (ne1, ne0) = (out, in).
    out_dim, in_dim = int(shp[1]), int(shp[0])
    return w.reshape(out_dim, in_dim)

layers = []
for L in range(Nl):
    p = f"blk.{L}."
    layers.append({
        "attn_norm": deq(p + "attn_norm.weight")[0],
        "ffn_norm":  deq(p + "ffn_norm.weight")[0],
        "Wq": load_w(p + "attn_q.weight"),
        "Wk": load_w(p + "attn_k.weight"),
        "Wv": load_w(p + "attn_v.weight"),
        "Wo": load_w(p + "attn_output.weight"),
        "Wg": load_w(p + "ffn_gate.weight"),
        "Wu": load_w(p + "ffn_up.weight"),
        "Wd": load_w(p + "ffn_down.weight"),
    })
print("FP weights loaded.", file=sys.stderr)

def rmsnorm(x, w):
    return x / np.sqrt((x * x).mean() + EPS) * w

def rope(v, pos, hd, theta):
    out = v.copy()
    for k in range(hd // 2):
        a = pos / (theta ** (2.0 * k / hd))
        c, s = math.cos(a), math.sin(a)
        out[:, 2*k]   = v[:, 2*k]*c - v[:, 2*k+1]*s
        out[:, 2*k+1] = v[:, 2*k]*s + v[:, 2*k+1]*c
    return out

def silu(x): return x / (1.0 + np.exp(-x))

def forward(tok, pos, kK, kV):
    h = tok_embd[tok].copy()
    for L in range(Nl):
        l = layers[L]
        x = rmsnorm(h, l["attn_norm"])
        q = (l["Wq"] @ x).reshape(Nh, Hd)
        k = (l["Wk"] @ x).reshape(Nkv, Hd)
        v = (l["Wv"] @ x).reshape(Nkv, Hd)
        q = rope(q, pos, Hd, THETA)
        k = rope(k, pos, Hd, THETA)
        kK[L, pos] = k; kV[L, pos] = v
        oh = np.zeros((Nh, Hd), dtype=np.float32)
        for hh in range(Nh):
            kvh = hh * Nkv // Nh
            sc = (kK[L, :pos+1, kvh] @ q[hh]) / math.sqrt(Hd)
            sc = np.exp(sc - sc.max()); sc /= sc.sum()
            oh[hh] = sc @ kV[L, :pos+1, kvh]
        h = h + l["Wo"] @ oh.reshape(Nh*Hd)
        x = rmsnorm(h, l["ffn_norm"])
        ff = silu(l["Wg"] @ x) * (l["Wu"] @ x)
        h = h + l["Wd"] @ ff
    x = rmsnorm(h, output_norm)
    return tok_embd @ x, h

kK = np.zeros((Nl, N_GEN+2, Nkv, Hd), dtype=np.float32)
kV = np.zeros((Nl, N_GEN+2, Nkv, Hd), dtype=np.float32)
tok, toks = 128000, []
for pos in range(N_GEN):
    logits, hid = forward(tok, pos, kK, kV)
    if pos == 0:
        top = np.argsort(logits)[-5:][::-1]
        print(f"pos=0 hidden[:6]={hid[:6]}", file=sys.stderr)
        print(f"pos=0 top5 ids={list(map(int,top))} "
              f"vals={[round(float(logits[i]),3) for i in top]}", file=sys.stderr)
    tok = int(logits.argmax()); toks.append(tok)
print("FP reference tokens:", toks)
print("Golden:             ", GOLDEN[:N_GEN])
print(f"match: {sum(1 for a,b in zip(toks,GOLDEN) if a==b)}/{len(toks)}")
