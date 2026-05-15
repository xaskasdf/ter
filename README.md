# ter

Balanced ternary CPU simulator + SIMD extension. Runs Llama 3.2 1B Q8_0,
BitNet b1.58 2B-4T, TinyStories Q4_K_M, and brandon-tiny end-to-end on a
ternary kernel substrate. CUDA backend ports the matmul fabric to RTX 3090,
where ternary-optimized packed kernels beat cuBLAS INT8 tensor cores by
1.65-1.90× without using tensor cores.

## TL;DR results

- **Llama 3.2 1B Q8_0** end-to-end forward on the ternary substrate: finite
  logits, op count parity 1.002× lane-MACs vs analytical fp16 baseline.
- **BitNet b1.58 2B-4T** end-to-end forward with i2_s native quantization,
  attn_sub_norm + ffn_sub_norm correctly wired.
- **CUDA packed-trit kernel** beats cuBLAS INT8 TC on RTX 3090 at the
  Llama 1B matmul fabric: 6.67 ms vs 12.67 ms = **1.90× faster**, with
  per-shape wins up to 4.18× (Wdown).
- **Memory footprint**: 8× compression vs fp16, 5× vs Q8_0; end-to-end
  Llama 1B forward needs 33% less VRAM than INT8 baseline.
- **Quality**: post-training BitNet quantization on Llama 1B Q8_0 (not
  trained for ternary) breaks quality (argmax flips, 0/5 top-5 overlap);
  Format A 11-trit (1+5+5) preserves argmax with RMSE 0.05.

Full results: `docs/baselines/2026-05-10-cuda-synthesis.md`.

## Build

    cmake -S . -B build
    cmake --build build
    ctest --test-dir build --output-on-failure

The matmul phase-gate test requires numpy. A local venv is provided at
`.venv/` (use `uv venv .venv && uv pip install --python .venv/bin/python numpy`
to recreate it).

The CUDA bench targets need a separate build with nvcc; see `cuda/`.

## Status (F0–F12)

### Foundation (F0–F4)
- [x] **F0** Trit, Tryte, Word27, Word54 primitives, packing, Memory.
- [x] **F1** Scalar ISA, assembler, simulator, sum(1..5) smoke test.
- [x] **F2** SIMD extension (tvadd, tvmac, tvsum, ...), 64×64 matmul gate.
- [x] **F3** Format B quantizer (bf16/float ↔ fixed-point trit + per-tensor scale).
- [x] **F4** All kernels (matmul, rmsnorm, softmax, silu, rope) + attention via
  host-orchestrated composition; complete K3 transformer building blocks.

### Inference bring-up (F5)
- [x] **F5.1–F5.3** Vendor ntransformer infra; single-layer forward; multi-token
  attention with KV cache; all 4 transcendental kernels plumbed.
- [x] **F5.4a–g** Brandon-tiny f16 GGUF loader, SPM tokenizer, value_residual,
  DWA, register prefill, multi-token generation.
- [x] **F5.4h** Calibration A/B (9-trit vs 12-trit greedy decoding).
- [x] **F5.4i** TinyStories Q4_K_M (Q4_K + Q6_K dequantizers).

### Llama 3.2 1B (F6)
- [x] **F6.1** Q8_0 dequantizer.
- [x] **F6.2** TritTensor payload Word27→int32 (27× memory reduction).
- [x] **F6.3** End-to-end Llama 1B forward, 508s/forward (pre-AVX2,
  kernel-routed mm_row), finite logits in [-10.29, 8.83]. Brought down to
  25.6s/forward by F11.
- [x] **F6.4** Op-count instrumentation; F8 honest TVMAC counter.
- [x] **F6.5** Multi-token generation (greedy from BOS).

### Hypothesis closure (F7–F9)
- [x] **F7** Format A (tfloat: 1t sign + 5t exp + N t mantissa); 11-trit (1+5+5)
  sweet spot preserves argmax on Llama 1B with RMSE 0.05.
- [x] **F8** Pure-scalar honest TVMAC counter (analytical bump for host-side ops).
- [x] **F9** BitNet b1.58 substrate-data alignment:
    - Analytical: TinyStories layer matmuls 121856 → 0 TVMAC (100% elimination)
    - Real GGUF: dequant_i2_s + 100% ternary purity verified on attn_q
    - Forward end-to-end: BitNet 2B-4T 18 min/forward, finite logits, sub_norm wired
    - Post-training quality: ternarizing Llama 1B Q8_0 (no QAT) breaks
      quality (argmax flips, 0/5 top-5 overlap, RMSE 2.39)

### Engineering (F10–F11)
- [x] **F10** K4 ring-0 backend (libter_k4.a):
    - C ABI freestanding-clean (-fno-exceptions -fno-rtti -DTER_FREESTANDING)
    - OpCounters → flat std::array<uint64_t, 256>
    - KernelTable → flat array (32 slots), no std::unordered_map
- [x] **F11** AVX2 GEMV in mm_row + -O3 on ter library: 21× speedup on Llama 1B
  (508 s/forward kernel-routed → 25.6 s/forward AVX2 GEMV).

### CUDA acceleration (F12)
- [x] **F12.1** mm_row CUDA naive: 21× speedup over Mac AVX2.
- [x] **F12.2** Microbench INT8 TC, sgemm fp32, naive packed (memory baseline).
- [x] **F12.3** End-to-end Llama 1B forward in CUDA (INT8 TC matmuls): 14.7 t/s.
- [x] **F12.4** Packed-trit end-to-end forward: 4.2 t/s, 33% memory savings.
- [x] **F12.5** RTX 3090 production baseline (llama.cpp Q8_0 Llama 3.2 1B, clean GPU): **395 t/s** reference (recalibrated 2026-05-15).
- [x] **F12.6** Real-weight quality validation on Llama 1B (BitNet post-quant breaks).
- [x] **F12.7** Ternary-optimized packed kernels:
    - v4 wide → v6 dp4a → v7 colmaj → v8 warp → v10 unroll → v11 warp4 → v13
    - Best-per-shape dispatch: **6.67 ms / 186 GMAC/s, 1.90× faster than INT8 TC**
- [x] **F12.8** End-to-end forward optimization (round 2): **14.7 → 421.5 t/s (28.7×)**
    - Fix 1: device-pointer scale eliminates 65 D2H syncs/token → 28.9 t/s
    - Fix 2: forward kernel upgrade v4 → v11 warp-coop → 200.6 t/s
    - Fix 3: cudaGraph capture (all state on-device, entire forward as one graph) → 421.5 t/s
    - Result: **1.07× faster than llama.cpp Q8_0** on Llama 1B, RTX 3090, 1.58 bits/weight, no tensor cores.

## Headline numbers (Llama 1B Q8_0, RTX 3090)

| Backend | ms / forward (matmul) | GMAC/s |
|---|---:|---:|
| Mac AVX2 sim (host scalar) | 25,600 | 38 |
| ter sim CUDA naive port | 26.06 | 47.5 |
| ter sim CUDA cuBLAS sgemm | 6.87 | 180 |
| **ter sim CUDA packed-trit (best dispatch)** | **6.67** | **186** |
| cuBLAS INT8 TC (production reference) | 12.67 | 98 |

Total session-cumulative speedup vs Mac AVX2 baseline: **3,840×**.
vs cuBLAS INT8 TC (binary substrate ceiling): **1.90× faster** without
tensor cores.

## What's proven vs what's engineering

### Architecturally proven (paper claims)
- Op count parity 1.002× lane-MACs vs fp16 (analytical).
- Memory compression 8× vs fp16, 5× vs Q8_0 (measured).
- Substrate-data alignment for BitNet: zero TVMAC at matmul fabric (analytical
  + real GGUF forward verified).
- Format A trit-budget tradeoff: 1+5+5 preserves quality on Llama 1B.
- Wall-clock superiority over INT8 TC at the matmul fabric level
  (1.90× faster on RTX 3090 with shape-aware kernel dispatch).

### Honest caveats
- End-to-end inference engine wall-clock vs llama.cpp Q8_0: not validated
  (a production-quality engine with kernel fusion + persistent attention is
  separate engineering work; the matmul fabric win is proven, the rest is
  engineering glue).
- Quality preservation requires the model to be trained for the target
  quantization (BitNet QAT). Post-training ternarization breaks quality.
- Per-op energy (J/MAC ternary CMOS vs binary CMOS): requires FPGA prototype
  to measure; the silicon-investment energy argument is theoretical without it.

## Building blocks

| Component | Purpose | File |
|---|---|---|
| `tk_matmul_b_9t` | 27-lane integer dot product (one tile of GEMM) | `src/kernels/tk_matmul_b_9t.tasm` |
| `tk_rmsnorm` | RMSNorm via rsqrt LUT | `src/kernels/tk_rmsnorm.tasm` |
| `tk_softmax` | Softmax via per-lane exp LUT + recip | `src/kernels/tk_softmax.tasm` |
| `tk_silu` | SiLU(x) = x · sigmoid(x); SwiGLU via host composition | `src/kernels/tk_silu.tasm` |
| `tk_rope` | Pure-SIMD paired rotation (host-prepared inputs) | `src/kernels/tk_rope.tasm` |
| `mm_row` (AVX2 GEMV) | Host-side matmul fabric, F8 honest TVMAC accounting | `src/tx/forward.cpp` |
| `mm_packed_v11/v13` (CUDA) | Ternary-optimized packed kernel, 4 cols per warp + dp4a | `cuda/ter_cuda_packed_v6.cu` |
| `forward_token` | End-to-end transformer forward (Llama, brandon, BitNet) | `src/tx/transformer.cpp` |
| `libter_k4.a` | C ABI for ring-0 osito-k integration | `src/k4/ter_k4.cpp`, `include/ter_k4/ter_k4.h` |

## Reference docs

- Design: `docs/superpowers/specs/2026-05-08-ter-design.md`
- Plans: `docs/superpowers/plans/`
- ISA: `docs/isa.md`
- Number formats (B, A): `docs/number-formats.md`
- Kernel patterns: `docs/kernel-patterns.md`
- RTX 3090 + CUDA full results: `docs/baselines/2026-05-10-cuda-synthesis.md`
- RTX 3090 production baseline (llama.cpp Q8_0): `docs/baselines/2026-05-10-rtx3090.md`
