#!/usr/bin/env python3
"""NumPy reference forward for Llama 3.2 1B using our converted packed binary.

Discriminates: bug in converter (Q8_0 dequant + ternarization) vs bug in
CUDA kernels. Loads /tmp/llama_3_2_1b_packed.bin (LLRT magic) and runs
greedy generation from BOS=128000.

Llama 3.2 1B vs BitNet differences:
  - No sub-norms (no attn_sub_norm, no ffn_sub_norm)
  - SiLU + mul FFN (not relu²)
  - NORM RoPE: rotates consecutive pairs (2k, 2k+1) — NOT NEOX
  - Tied lm_head: token_embd reused for output projection (fp16 matmul)

If output matches Llama Q8_0 golden tokens
  [28783, 25, 220, 679, 24, 11, 220, 2366, ...] = " Tags: 2019, 2020, ..."
-> ternarization preserves quality (unlikely per H1).
If output diverges from Q8_0 golden but our CUDA forward matches THIS
reference -> kernel correct, ternarization is the model-quality story.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import struct
import math

BIN = '/tmp/llama_3_2_1b_packed.bin'
N_GEN = 16

with open(BIN, 'rb') as f:
    raw_header = f.read(64)
    magic, version = struct.unpack('<II', raw_header[:8])
    assert magic == 0x4C4C5254 and version == 1, f"bad magic {hex(magic)}"
    H, F, Nl, V, Nh, Nkv, Hd, Smax = struct.unpack('<IIIIIIII', raw_header[8:40])
    rope_theta, eps = struct.unpack('<ff', raw_header[40:48])
    print(f"Config: H={H} F={F} Nl={Nl} V={V} Nh={Nh} Nkv={Nkv} "
          f"Hd={Hd} rope={rope_theta} eps={eps}", file=sys.stderr)

    Hkv = Nkv * Hd
    tok_embd_bytes = V * H * 2
    tok_embd = (np.frombuffer(f.read(tok_embd_bytes), dtype=np.float16)
                  .reshape(V, H).astype(np.float32))
    output_norm = np.frombuffer(f.read(H * 4), dtype=np.float32)

    layers = []
    for L in range(Nl):
        attn_norm = np.frombuffer(f.read(H * 4), dtype=np.float32)
        ffn_norm = np.frombuffer(f.read(H * 4), dtype=np.float32)
        weights = {}
        for nm, K_in, N_out in [
            ('attn_q',      H, H),
            ('attn_k',      H, Hkv),
            ('attn_v',      H, Hkv),
            ('attn_output', H, H),
            ('ffn_gate',    H, F),
            ('ffn_up',      H, F),
            ('ffn_down',    F, H),
        ]:
            bytes_n = K_in * (N_out // 4)
            packed = (np.frombuffer(f.read(bytes_n), dtype=np.uint8)
                        .reshape(N_out // 4, K_in).copy())
            scale = struct.unpack('<f', f.read(4))[0]
            # Decode packed -> ternary [N_out, K_in]: code 0=zero, 1=+1, 2=-1.
            codes = np.empty((N_out, K_in), dtype=np.int8)
            for sub in range(4):
                bits = (packed >> (sub * 2)) & 3
                vals = np.where(bits == 1, 1, np.where(bits == 2, -1, 0))
                codes[sub::4, :] = vals
            weights[nm] = (codes.astype(np.float32), scale)
        layers.append({
            'attn_norm': attn_norm, 'ffn_norm': ffn_norm, 'w': weights,
        })

print("Weights loaded.", file=sys.stderr)


def rmsnorm(x, w, eps_=eps):
    rms = np.sqrt((x * x).mean() + eps_)
    return x / rms * w


def quant_int8(x):
    amax = max(float(np.abs(x).max()), 1e-9)
    scale = amax / 127.0
    q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return q, scale


def matmul_packed(x_q, W_codes, scale_x, w_scale):
    """W_codes shape [N, K] (each row is one output col's K-vec)."""
    acc = W_codes.astype(np.int32) @ x_q.astype(np.int32)
    return acc.astype(np.float32) * (scale_x * w_scale)


def rope_norm(v, pos, head_dim, theta):
    """Standard Llama RoPE (NORM, not NEOX): rotate consecutive pairs (2k, 2k+1)."""
    half = head_dim // 2
    out = v.copy()
    for k in range(half):
        freq = 1.0 / (theta ** (2.0 * k / head_dim))
        a = pos * freq
        c, s = math.cos(a), math.sin(a)
        v0 = v[:, 2 * k]; v1 = v[:, 2 * k + 1]
        out[:, 2 * k]     = v0 * c - v1 * s
        out[:, 2 * k + 1] = v0 * s + v1 * c
    return out


def silu(x):
    return x / (1.0 + np.exp(-x))


def forward(token_id, pos, kv_K, kv_V):
    hidden = tok_embd[token_id].copy().astype(np.float32)
    for L in range(Nl):
        l = layers[L]
        # ----- attention -----
        x = rmsnorm(hidden, l['attn_norm'])
        x_q, sx = quant_int8(x)
        Wq, sWq = l['w']['attn_q']
        Wk, sWk = l['w']['attn_k']
        Wv, sWv = l['w']['attn_v']
        q = matmul_packed(x_q, Wq, sx, sWq).reshape(Nh, Hd)
        k = matmul_packed(x_q, Wk, sx, sWk).reshape(Nkv, Hd)
        v = matmul_packed(x_q, Wv, sx, sWv).reshape(Nkv, Hd)
        q = rope_norm(q, pos, Hd, rope_theta)
        k = rope_norm(k, pos, Hd, rope_theta)
        kv_K[L, pos] = k
        kv_V[L, pos] = v
        # GQA: each q-head h reads kv head h * Nkv / Nh
        out_heads = np.zeros((Nh, Hd), dtype=np.float32)
        inv_sqrt = 1.0 / math.sqrt(Hd)
        for h in range(Nh):
            kvh = h * Nkv // Nh
            scores = (kv_K[L, :pos + 1, kvh] @ q[h]) * inv_sqrt   # [pos+1]
            scores = scores - scores.max()
            scores = np.exp(scores); scores /= scores.sum()
            out_heads[h] = scores @ kv_V[L, :pos + 1, kvh]        # [Hd]
        attn_out = out_heads.reshape(Nh * Hd)
        x_q, sx = quant_int8(attn_out)
        Wo, sWo = l['w']['attn_output']
        attn_proj = matmul_packed(x_q, Wo, sx, sWo)
        hidden = hidden + attn_proj

        # ----- FFN -----
        x = rmsnorm(hidden, l['ffn_norm'])
        x_q, sx = quant_int8(x)
        Wg, sWg = l['w']['ffn_gate']
        Wu, sWu = l['w']['ffn_up']
        gate = matmul_packed(x_q, Wg, sx, sWg)
        up   = matmul_packed(x_q, Wu, sx, sWu)
        ff = silu(gate) * up
        f_q, sf = quant_int8(ff)
        Wd, sWd = l['w']['ffn_down']
        ff_proj = matmul_packed(f_q, Wd, sf, sWd)
        hidden = hidden + ff_proj

    # output_norm + tied lm_head (fp16 matmul against token_embd)
    x = rmsnorm(hidden, output_norm)
    logits = tok_embd @ x  # [V, H] @ [H] -> [V]
    return logits, hidden


print(f"Greedy from BOS=128000, n_gen={N_GEN}...", file=sys.stderr)
kv_K = np.zeros((Nl, Smax, Nkv, Hd), dtype=np.float32)
kv_V = np.zeros((Nl, Smax, Nkv, Hd), dtype=np.float32)

tok = 128000
toks = []
for pos in range(N_GEN):
    logits, hidden = forward(tok, pos, kv_K, kv_V)
    tok = int(logits.argmax())
    toks.append(tok)
    if pos == 0:
        # Diagnostics
        print(f"  pos=0: hidden range=[{hidden.min():.3f}, {hidden.max():.3f}] "
              f"std={hidden.std():.3f}", file=sys.stderr)
        print(f"  pos=0: top-5 logits ids="
              f"{list(np.argsort(logits)[-5:][::-1])} "
              f"vals={[f'{logits[i]:.3f}' for i in np.argsort(logits)[-5:][::-1]]}",
              file=sys.stderr)

print(f"\nTernarized Llama 1B greedy tokens (n={len(toks)}):")
print(toks)
print()
print(f"Q8_0 golden reference (llama-cli):")
print([28783, 25, 220, 679, 24, 11, 220, 2366, 15, 11, 220, 2366, 16, 11, 220, 2366][:N_GEN])
match = sum(1 for a, b in zip(toks, [28783, 25, 220, 679, 24, 11, 220, 2366,
                                      15, 11, 220, 2366, 16, 11, 220, 2366][:N_GEN])
            if a == b)
print(f"Tokens matching Q8_0: {match}/{N_GEN} ({100.0*match/N_GEN:.0f}%)")
