#!/usr/bin/env python3
"""Convert BitNet 2B-4T i2_s GGUF to a flat binary blob for our CUDA forward.

Output layout (little-endian):
  Header (64 bytes):
    magic     uint32  = 0x54455254  ('TERT')
    version   uint32  = 1
    H, F, Nl, V, Nh, Nkv, Hd  uint32 each (28 bytes)
    Smax      uint32  = 64
    rope_theta float32
    eps       float32
    padding to 64

  token_embd_f16:        V * H halves  (V rows of H halves)
  output_norm_f32:       H floats

  Per layer L in [0, Nl):
    attn_norm_f32:       H floats
    ffn_norm_f32:        H floats
    attn_sub_norm_f32:   H floats
    ffn_sub_norm_f32:    F floats
    Wq_packed:           H * (H/4) bytes        (col-major: 4 cols per byte, K=H rows)
    Wk_packed:           H * (Hkv/4) bytes
    Wv_packed:           H * (Hkv/4) bytes
    Wo_packed:           H * (H/4) bytes
    Wgate_packed:        H * (F/4) bytes
    Wup_packed:          H * (F/4) bytes
    Wdown_packed:        F * (H/4) bytes

Col-major packed-trit layout matches v11 kernel: W_col[j_byte * K + k]
holds 4 codes at output cols 4*j_byte .. 4*j_byte+3, input row k.
"""

import gguf, gguf.constants as gc, gguf.gguf_reader as ggr
import numpy as np
import enum
import struct
import sys

# Monkey-patch I2_S = 36
new = {m.name: m.value for m in gc.GGMLQuantizationType}
new['I2_S'] = 36
NE = enum.IntEnum('GGMLQuantizationType', new)
gc.GGMLQuantizationType = NE
gguf.GGMLQuantizationType = NE
ggr.GGMLQuantizationType = NE
gc.GGML_QUANT_SIZES[NE.I2_S] = (4, 1)
ggr.GGML_QUANT_SIZES = gc.GGML_QUANT_SIZES

GGUF_PATH = '/Users/pc/osito-a-models/downloads/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf'
OUT_PATH = '/tmp/bitnet_2b_4t_packed.bin'

print(f"Loading {GGUF_PATH}...")
r = gguf.GGUFReader(GGUF_PATH)

# Config
def kv_u32(name):
    f = r.fields[name]
    return int(f.parts[f.data[0]][0])
def kv_f32(name):
    f = r.fields[name]
    return float(f.parts[f.data[0]][0])

H = kv_u32('bitnet-b1.58.embedding_length')
F = kv_u32('bitnet-b1.58.feed_forward_length')
Nl = kv_u32('bitnet-b1.58.block_count')
V = kv_u32('bitnet-b1.58.vocab_size')
Nh = kv_u32('bitnet-b1.58.attention.head_count')
Nkv = kv_u32('bitnet-b1.58.attention.head_count_kv')
rope_dim = kv_u32('bitnet-b1.58.rope.dimension_count')
Hd = H // Nh  # head_dim
assert Hd == rope_dim, f"head_dim mismatch: H/Nh={Hd} vs rope_dim={rope_dim}"
eps = kv_f32('bitnet-b1.58.attention.layer_norm_rms_epsilon')
rope_theta = kv_f32('bitnet-b1.58.rope.freq_base')
Smax = 64

print(f"Config: H={H} F={F} Nl={Nl} Nh={Nh} Nkv={Nkv} Hd={Hd} V={V} eps={eps} rope={rope_theta}")

def get_tensor(name):
    for t in r.tensors:
        if t.name == name:
            return t
    raise KeyError(name)

def read_i2s_scale(t, fp):
    """microsoft/BitNet's GGUF i2_s format appends 8 fp32 values (32 bytes)
    after each tensor's codes. The first fp32 is the per-tensor scale
    (microsoft's `i2_scale`). The remaining 28 bytes are padding/garbage."""
    fp.seek(t.data_offset + t.n_bytes)
    raw = fp.read(4)
    return struct.unpack('<f', raw)[0]

def i2s_to_col_packed(t):
    """Convert microsoft/BitNet i2_s tensor to our v11 col-major packed-trit.

    Microsoft's actual layout (ggml-quants.c dequantize_row_i2_s, line 71 of
    ggml-bitnet-mad.cpp's quantize_i2_s ACT_PARALLEL branch with QK_I2_S=128):
      - Tensor is processed in flat row-major order in BLOCKS of 128 elements.
      - Each 128-element block packs into 32 consecutive bytes.
      - Byte p in block (p=0..31) holds 4 codes from element positions
        {p, p+32, p+64, p+96} within the block.
      - Bit order within byte: position 0 (group 0) at HIGH bits (shift 6),
        position 96 (group 3) at LOW bits (shift 0).
      - Code map: 0->-1, 1->0, 2->+1, 3->0.

    Conversion:
      1. Read raw bytes (total = N*K/4 bytes)
      2. Reshape blocks of 32 bytes, decode 4 codes per byte to recover the
         flat (N, K) tensor in row-major.
      3. Remap codes to our format (0=0, 1=+1, 2=-1).
      4. Re-pack to our v11 col-major: out[j_byte, k] holds 4 output cols
         {4*j_byte..4*j_byte+3} at input row k, low bits = first col.
    """
    assert int(t.tensor_type) == 36, f"expected i2_s, got {t.tensor_type}"
    K = int(t.shape[0])
    N = int(t.shape[1])
    assert K % 4 == 0 and N % 4 == 0
    raw = np.array(t.data, copy=False).view(np.uint8)
    total = N * K
    assert raw.size == total // 4, f"bytes {raw.size} != {total//4}"

    # Decode flat[N*K] from raw[total/4] bytes
    # Group bytes into 32-byte blocks; each block represents 128 elements
    n_blocks = total // 128
    raw_blocks = raw.reshape(n_blocks, 32)  # n_blocks of 32 bytes each

    # Each byte p in a block holds 4 codes for elements at positions p, p+32, p+64, p+96
    # of that block. Shifts: group 0 (positions 0..31) at HIGH bits (shift 6).
    flat_codes = np.empty(total, dtype=np.uint8)
    for group in range(4):
        shift = 6 - group * 2
        # codes[block, p] -> flat position block*128 + group*32 + p
        decoded = (raw_blocks >> shift) & 3
        # Reshape and place
        flat_codes[group*32::128] = 0  # init slot positions across all blocks
    # Better: vectorize properly
    # For each block, for each group, positions block*128 + group*32 + (0..31)
    for group in range(4):
        shift = 6 - group * 2
        decoded = (raw_blocks >> shift) & 3  # shape (n_blocks, 32)
        # flat positions: each block contributes 32 consecutive elements at offset group*32
        # block b, position p in group -> flat = b*128 + group*32 + p
        for b in range(n_blocks):
            flat_codes[b*128 + group*32 : b*128 + group*32 + 32] = decoded[b]

    # Tensor in (N, K) row-major (each row is K consecutive elements)
    codes = flat_codes.reshape(N, K)

    # Remap microsoft -> ours
    remap = np.array([2, 0, 1, 0], dtype=np.uint8)
    codes = remap[codes]

    # Pack into our v11 col-major
    out = np.zeros((N // 4, K), dtype=np.uint8)
    for sub in range(4):
        out |= (codes[sub::4, :] & 0x3) << (sub * 2)
    return out.flatten()  # shape: (N/4) * K bytes total

print(f"Writing {OUT_PATH}...")
# Need separate raw file handle to read scales (per-tensor 32-byte trailer)
src_fp = open(GGUF_PATH, 'rb')
with open(OUT_PATH, 'wb') as f:
    # Header: 64 bytes
    f.write(struct.pack('<II', 0x54455254, 1))  # magic, version
    f.write(struct.pack('<IIIIIIII', H, F, Nl, V, Nh, Nkv, Hd, Smax))
    f.write(struct.pack('<ff', rope_theta, eps))
    f.write(b'\x00' * (64 - 8 - 32 - 8))  # pad to 64

    # Token embd (F16)
    t = get_tensor('token_embd.weight')
    print(f"  token_embd: {t.shape} {t.tensor_type} -> {t.n_bytes} bytes")
    data = np.array(t.data, copy=False).view(np.uint8)
    assert data.size == V * H * 2, f"token_embd size {data.size} != {V*H*2}"
    f.write(data.tobytes())

    # output_norm (F32)
    t = get_tensor('output_norm.weight')
    data = np.array(t.data, copy=False).astype(np.float32)
    assert data.size == H
    f.write(data.tobytes())

    Hkv = Nkv * Hd
    for L in range(Nl):
        for nm, expect_size in [
            ('attn_norm', H),
            ('ffn_norm', H),
            ('attn_sub_norm', H),
            ('ffn_sub_norm', F),
        ]:
            t = get_tensor(f'blk.{L}.{nm}.weight')
            data = np.array(t.data, copy=False).astype(np.float32)
            assert data.size == expect_size, f"{nm}@L={L}: {data.size} != {expect_size}"
            f.write(data.tobytes())

        for nm, K_in, N_out in [
            ('attn_q',      H, H),
            ('attn_k',      H, Hkv),
            ('attn_v',      H, Hkv),
            ('attn_output', H, H),
            ('ffn_gate',    H, F),
            ('ffn_up',      H, F),
            ('ffn_down',    F, H),
        ]:
            t = get_tensor(f'blk.{L}.{nm}.weight')
            packed = i2s_to_col_packed(t)
            scale = read_i2s_scale(t, src_fp)
            assert packed.size == K_in * (N_out // 4), \
                f"{nm}@L={L}: packed {packed.size} != {K_in * (N_out // 4)}"
            # Write packed weights then fp32 scale for this tensor
            f.write(packed.tobytes())
            f.write(struct.pack('<f', scale))

        if L % 5 == 0:
            print(f"  layer {L}/{Nl} done")

print(f"Done. Wrote {OUT_PATH}")
import os
sz = os.path.getsize(OUT_PATH)
print(f"Size: {sz} bytes = {sz/1024/1024:.1f} MB")
