#!/usr/bin/env python3
"""Convert Llama 3.2 1B Q8_0 GGUF to a flat binary blob for our CUDA forward.

Ternarizes each Q8_0 weight tensor to BitNet b1.58 ({-1, 0, +1}) using the
per-tensor recipe:
    scale = mean(|x|)
    q     = round(x / scale) clamped to {-1, 0, +1}
then packs into the v11 col-major layout that mm_packed_dispatch expects:
    W[j_byte, k] holds 4 codes for output cols {4*j_byte..4*j_byte+3}
    at input row k. Low bits = first col.

Output layout (little-endian):
  Header (64 bytes):
    magic     uint32  = 0x4C4C5254  ('LLRT')
    version   uint32  = 1
    H, F, Nl, V, Nh, Nkv, Hd  uint32 each (28 bytes)
    Smax      uint32
    rope_theta float32
    eps       float32
    padding to 64

  token_embd_f16:        V * H halves (fp16)  -- also serves as tied lm_head
  output_norm_f32:       H floats

  Per layer L in [0, Nl):
    attn_norm_f32:       H floats
    ffn_norm_f32:        H floats
    Wq_packed + scale:   H * (H/4) bytes + 4 bytes scale (fp32)
    Wk_packed + scale:   H * (Hkv/4) + 4
    Wv_packed + scale:   H * (Hkv/4) + 4
    Wo_packed + scale:   H * (H/4) + 4
    Wgate_packed + scale: H * (F/4) + 4
    Wup_packed + scale:  H * (F/4) + 4
    Wdown_packed + scale: F * (H/4) + 4

Notes:
  - Llama 3.2 1B has tied weights: no separate output.weight; lm_head reuses
    token_embd. The CUDA loader must dispatch to an fp16 lm_head matmul
    against m.token_embd (mirroring the BitNet path), not a packed-trit
    matmul against the (already-ternarized) per-layer weights.
  - GGUF Q8_0 storage is (N, bytes_per_row) raw; gguf.dequantize returns
    (N, K) fp32. Each row is one output column's K-vector.
"""

import gguf
import numpy as np
import struct
import os
import sys

GGUF_PATH = (sys.argv[1] if len(sys.argv) > 1
             else '/Users/pc/osito-a-models/downloads/llama-3.2-1b-instruct/'
                  'llama-3.2-1b-instruct-q8_0.gguf')
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else '/tmp/llama_3_2_1b_packed.bin'

print(f"Loading {GGUF_PATH}...")
r = gguf.GGUFReader(GGUF_PATH)


def kv_u32(name):
    f = r.fields[name]
    return int(f.parts[f.data[0]][0])


def kv_f32(name):
    f = r.fields[name]
    return float(f.parts[f.data[0]][0])


H = kv_u32('llama.embedding_length')
F = kv_u32('llama.feed_forward_length')
Nl = kv_u32('llama.block_count')
V = kv_u32('llama.vocab_size')
Nh = kv_u32('llama.attention.head_count')
Nkv = kv_u32('llama.attention.head_count_kv')
rope_dim = kv_u32('llama.rope.dimension_count')
Hd = H // Nh
assert Hd == rope_dim, f"head_dim mismatch: H/Nh={Hd} vs rope_dim={rope_dim}"
eps = kv_f32('llama.attention.layer_norm_rms_epsilon')
rope_theta = kv_f32('llama.rope.freq_base')
Smax = 64
Hkv = Nkv * Hd

print(f"Config: H={H} F={F} Nl={Nl} Nh={Nh} Nkv={Nkv} Hd={Hd} V={V} "
      f"eps={eps} rope_theta={rope_theta} Hkv={Hkv}")


def get_tensor(name):
    for t in r.tensors:
        if t.name == name:
            return t
    raise KeyError(name)


def q8_0_to_fp32(t):
    """Dequantize a Q8_0 tensor to fp32. Returns (N, K) where N=output cols
    and K=input dim. Each row is one output column's K-vector.
    """
    raw = np.array(t.data, copy=False)
    fp32 = gguf.dequantize(raw, t.tensor_type)
    K = int(t.shape[0])
    N = int(t.shape[1])
    assert fp32.shape == (N, K), \
        f"unexpected shape {fp32.shape} vs ({N}, {K}) for {t.name}"
    return fp32.astype(np.float32, copy=False), K, N


def ternarize_to_codes(fp32):
    """BitNet b1.58 per-tensor ternarization.
        scale = mean(|x|)
        q     = clip(round(x / scale), -1, +1)
    Returns (codes_uint8 in {0=0, 1=+1, 2=-1}, scale_float).
    """
    abs_mean = float(np.mean(np.abs(fp32)))
    if abs_mean == 0.0:
        codes = np.zeros(fp32.size, dtype=np.uint8)
        return codes, 0.0
    q = np.rint(fp32 / abs_mean)
    np.clip(q, -1, 1, out=q)
    # Map: -1 -> 2, 0 -> 0, +1 -> 1 (matches BitNet packed kernel decoding:
    # (b >> 2*sub) & 3 -> 1 for +x, 2 for -x, 0 to skip).
    q_int = q.astype(np.int8)
    codes = np.where(q_int == 0, 0,
            np.where(q_int > 0, 1, 2)).astype(np.uint8)
    return codes, abs_mean


def pack_col_major_v11(codes_flat, K, N):
    """Pack flat codes (length N*K, row-major (N, K)) into v11 col-major
    layout: out[j_byte, k] = 4 codes for cols {4*j_byte..4*j_byte+3} at
    input row k. Low bits = first col.

    Returns flat byte array of size (N//4) * K.
    """
    assert N % 4 == 0
    codes_NK = codes_flat.reshape(N, K)
    out = np.zeros((N // 4, K), dtype=np.uint8)
    for sub in range(4):
        out |= (codes_NK[sub::4, :] & 0x3) << (sub * 2)
    return out.reshape(-1)


def encode_packed_weight(t, expect_K, expect_N):
    fp32, K, N = q8_0_to_fp32(t)
    assert (K, N) == (expect_K, expect_N), \
        f"{t.name}: dequant K,N=({K},{N}) != expected ({expect_K},{expect_N})"
    codes, scale = ternarize_to_codes(fp32)
    packed = pack_col_major_v11(codes, K, N)
    nz_pct = 100.0 * float(np.mean(codes != 0))
    return packed, scale, nz_pct


print(f"Writing {OUT_PATH}...")
with open(OUT_PATH, 'wb') as f:
    # Header (64 bytes)
    f.write(struct.pack('<II', 0x4C4C5254, 1))             # magic, version
    f.write(struct.pack('<IIIIIIII', H, F, Nl, V, Nh, Nkv, Hd, Smax))
    f.write(struct.pack('<ff', rope_theta, eps))
    f.write(b'\x00' * (64 - 8 - 32 - 8))                   # pad to 64

    # token_embd: Q8_0 -> fp16 (will be used as both embed lookup AND tied lm_head)
    t = get_tensor('token_embd.weight')
    fp32, K, N = q8_0_to_fp32(t)   # N=V, K=H
    assert (K, N) == (H, V)
    fp16 = fp32.astype(np.float16)
    f.write(fp16.tobytes())
    print(f"  token_embd: {V} x {H} fp16 = {fp16.nbytes/1024/1024:.1f} MB")

    # output_norm (F32)
    t = get_tensor('output_norm.weight')
    on = np.array(t.data, copy=False).astype(np.float32)
    assert on.size == H, f"output_norm size {on.size} != {H}"
    f.write(on.tobytes())

    # Per-layer
    for L in range(Nl):
        # F32 norms
        for nm, expect_size in [('attn_norm', H), ('ffn_norm', H)]:
            tn = get_tensor(f'blk.{L}.{nm}.weight')
            data = np.array(tn.data, copy=False).astype(np.float32)
            assert data.size == expect_size, \
                f"{nm}@L={L}: {data.size} != {expect_size}"
            f.write(data.tobytes())

        # Packed-trit weights with per-tensor scale
        layer_nz = []
        layer_scales = []
        for nm, K_in, N_out in [
            ('attn_q',      H, H),
            ('attn_k',      H, Hkv),
            ('attn_v',      H, Hkv),
            ('attn_output', H, H),
            ('ffn_gate',    H, F),
            ('ffn_up',      H, F),
            ('ffn_down',    F, H),
        ]:
            tw = get_tensor(f'blk.{L}.{nm}.weight')
            packed, scale, nz_pct = encode_packed_weight(tw, K_in, N_out)
            assert packed.size == K_in * (N_out // 4), \
                f"{nm}@L={L}: packed {packed.size} != {K_in*(N_out//4)}"
            f.write(packed.tobytes())
            f.write(struct.pack('<f', scale))
            layer_nz.append(nz_pct)
            layer_scales.append(scale)

        if L % 4 == 0 or L == Nl - 1:
            avg_nz = sum(layer_nz) / len(layer_nz)
            print(f"  layer {L:2d}/{Nl}: avg_nz={avg_nz:5.1f}% "
                  f"scales={['%.4f' % s for s in layer_scales]}")

print(f"Done. Wrote {OUT_PATH}")
sz = os.path.getsize(OUT_PATH)
print(f"Size: {sz} bytes = {sz/1024/1024:.1f} MB "
      f"(expected ~{(V*H*2 + H*4 + Nl*(H*2*4 + (4*H*H + 2*H*Hkv + 2*H*F + F*H)//4 + 7*4))/1024/1024:.0f} MB)")
