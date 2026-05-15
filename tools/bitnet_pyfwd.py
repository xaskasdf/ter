#!/usr/bin/env python3
"""Numpy BitNet 2B-4T forward using our converted packed binary.
Discriminates: bug in converter (i2_s decode) vs bug in CUDA kernels.

Loads /tmp/bitnet_2b_4t_packed.bin which has our col-major packed-trit
format + scales + norm gammas + token_embd. Runs greedy generation
from BOS and prints first 8 tokens.

If output matches microsoft reference [11, 220, 16, 11, 220, 17, ...]
-> our converter is CORRECT, bug is in CUDA forward.
If output doesn't match -> converter has subtle bug, fix that first.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import struct
import math

BIN = '/tmp/bitnet_2b_4t_packed.bin'

with open(BIN, 'rb') as f:
    raw_header = f.read(64)
    magic, version = struct.unpack('<II', raw_header[:8])
    assert magic == 0x54455254 and version == 1
    H, F, Nl, V, Nh, Nkv, Hd, Smax = struct.unpack('<IIIIIIII', raw_header[8:40])
    rope_theta, eps = struct.unpack('<ff', raw_header[40:48])
    print(f"Config: H={H} F={F} Nl={Nl} V={V} Nh={Nh} Nkv={Nkv} Hd={Hd}", file=sys.stderr)

    # Token embedding (V * H * 2 bytes, fp16)
    tok_embd_bytes = V * H * 2
    tok_embd = np.frombuffer(f.read(tok_embd_bytes), dtype=np.float16).reshape(V, H).astype(np.float32)
    # output_norm (H floats)
    output_norm = np.frombuffer(f.read(H * 4), dtype=np.float32)

    layers = []
    Hkv = Nkv * Hd
    for L in range(Nl):
        attn_norm = np.frombuffer(f.read(H * 4), dtype=np.float32)
        ffn_norm = np.frombuffer(f.read(H * 4), dtype=np.float32)
        attn_sub_norm = np.frombuffer(f.read(H * 4), dtype=np.float32)
        ffn_sub_norm = np.frombuffer(f.read(F * 4), dtype=np.float32)
        weights = {}
        for nm, K_in, N_out in [
            ('attn_q', H, H), ('attn_k', H, Hkv), ('attn_v', H, Hkv), ('attn_output', H, H),
            ('ffn_gate', H, F), ('ffn_up', H, F), ('ffn_down', F, H),
        ]:
            bytes_n = K_in * (N_out // 4)
            packed = np.frombuffer(f.read(bytes_n), dtype=np.uint8).reshape(N_out // 4, K_in).copy()
            scale = struct.unpack('<f', f.read(4))[0]
            # Decode packed to ternary {-1, 0, +1} matrix [N_out, K_in]
            codes = np.empty((N_out, K_in), dtype=np.int8)
            for sub in range(4):
                bits = (packed >> (sub * 2)) & 3
                # our code: 0=zero, 1=+1, 2=-1
                vals = np.where(bits == 1, 1, np.where(bits == 2, -1, 0))
                codes[sub::4, :] = vals
            weights[nm] = (codes.astype(np.float32), scale)
        layers.append({'attn_norm': attn_norm, 'ffn_norm': ffn_norm,
                       'attn_sub_norm': attn_sub_norm, 'ffn_sub_norm': ffn_sub_norm,
                       'w': weights})
        if L == 0:
            print(f"  layer 0 attn_norm mean={attn_norm.mean():.4f} std={attn_norm.std():.4f}", file=sys.stderr)
            print(f"  layer 0 Wq scale={weights['attn_q'][1]:.4f}, codes mean={weights['attn_q'][0].mean():.4f}",
                  file=sys.stderr)

print("Weights loaded.", file=sys.stderr)

def rmsnorm(x, w, eps=1e-5):
    rms = np.sqrt((x*x).mean() + eps)
    return x / rms * w

def quant_int8(x):
    amax = max(np.abs(x).max(), 1e-9)
    scale = amax / 127.0
    q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return q, scale

def matmul_packed(x_q, W_codes, scale_x, w_scale):
    # x_q [K] int8, W_codes [N, K] in {-1, 0, +1}, returns [N]
    acc = W_codes.astype(np.int32) @ x_q.astype(np.int32)
    return acc.astype(np.float32) * (scale_x * w_scale)

def rope_neox(v, pos, head_dim, theta):
    # v shape [n_heads, head_dim]
    half = head_dim // 2
    out = v.copy()
    for k in range(half):
        freq = 1.0 / (theta ** (2.0 * k / head_dim))
        a = pos * freq
        c, s = math.cos(a), math.sin(a)
        v0 = v[:, k]; v1 = v[:, k + half]
        out[:, k] = v0 * c - v1 * s
        out[:, k + half] = v0 * s + v1 * c
    return out

def forward(token_id, pos, kv_cache_K, kv_cache_V):
    hidden = tok_embd[token_id].copy()
    for L in range(Nl):
        l = layers[L]
        # attention block
        x_norm = rmsnorm(hidden, l['attn_norm'], eps)
        x_q, scale_x = quant_int8(x_norm)
        wq, sq = l['w']['attn_q']
        wk, sk = l['w']['attn_k']
        wv, sv = l['w']['attn_v']
        q = matmul_packed(x_q, wq, scale_x, sq).reshape(Nh, Hd)
        k = matmul_packed(x_q, wk, scale_x, sk).reshape(Nkv, Hd)
        v = matmul_packed(x_q, wv, scale_x, sv).reshape(Nkv, Hd)
        # RoPE NEOX
        q = rope_neox(q, pos, Hd, rope_theta)
        k = rope_neox(k, pos, Hd, rope_theta)
        # KV cache store
        kv_cache_K[L, pos] = k
        kv_cache_V[L, pos] = v
        # attention
        attn_out = np.zeros(H, dtype=np.float32)
        for h in range(Nh):
            kv_h = h * Nkv // Nh
            scores = np.zeros(pos + 1, dtype=np.float32)
            for s in range(pos + 1):
                scores[s] = float(q[h] @ kv_cache_K[L, s, kv_h]) / math.sqrt(Hd)
            mx = scores.max()
            exps = np.exp(scores - mx)
            scores = exps / exps.sum()
            for s in range(pos + 1):
                attn_out[h * Hd:(h+1) * Hd] += scores[s] * kv_cache_V[L, s, kv_h]
        # attn_sub_norm + Wo
        x_norm = rmsnorm(attn_out, l['attn_sub_norm'], eps)
        x_q, scale_x = quant_int8(x_norm)
        wo, so = l['w']['attn_output']
        attn_proj = matmul_packed(x_q, wo, scale_x, so)
        hidden = hidden + attn_proj
        # FFN
        x_norm = rmsnorm(hidden, l['ffn_norm'], eps)
        x_q, scale_x = quant_int8(x_norm)
        wgate, sg = l['w']['ffn_gate']
        wup, su = l['w']['ffn_up']
        gate = matmul_packed(x_q, wgate, scale_x, sg)
        up = matmul_packed(x_q, wup, scale_x, su)
        ff = np.where(gate > 0, gate * gate, 0) * up
        ff_norm = rmsnorm(ff, l['ffn_sub_norm'], eps)
        ff_q, ff_scale = quant_int8(ff_norm)
        wdown, sd = l['w']['ffn_down']
        ffn_proj = matmul_packed(ff_q, wdown, ff_scale, sd)
        hidden = hidden + ffn_proj
        if L < 2:
            print(f"  L={L} hidden mean={hidden.mean():.4f} std={hidden.std():.4f}",
                  file=sys.stderr)
    # output_norm + lm_head (tied to token_embd)
    x_norm = rmsnorm(hidden, output_norm, eps)
    logits = tok_embd @ x_norm  # [V]
    return logits, hidden

# Generate
kv_K = np.zeros((Nl, Smax, Nkv, Hd), dtype=np.float32)
kv_V = np.zeros((Nl, Smax, Nkv, Hd), dtype=np.float32)

token = 128000  # BOS
print(f"\nStarting greedy from BOS=128000:", file=sys.stderr)
tokens = [token]
for pos in range(8):
    logits, hidden = forward(token, pos, kv_K, kv_V)
    # top-5
    top_idx = np.argsort(-logits)[:5]
    top_logits = logits[top_idx]
    if pos == 0:
        print(f"  pos=0 top-5: {list(zip(top_idx.tolist(), top_logits.tolist()))}", file=sys.stderr)
    token = int(top_idx[0])
    tokens.append(token)
print(f"\nNumpy forward tokens: {tokens}")
print(f"Reference (BOS only):   [128000, 11, 220, 16, 11, 220, 17, 11, 220]")
