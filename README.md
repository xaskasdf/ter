# ter

Balanced-ternary substrate simulator (F0–F11) that runs Llama 3.2 1B
faster than llama.cpp Q8_0 and reproduces BitNet b1.58 2B-4T bit-exact
at 1.58 bits/weight on consumer GPUs. The substrate simulator is the
research vehicle; the CUDA port is an iteration aid for the matmul
fabric.

## TL;DR results (two engineering claims, defendable end-to-end)

- **Llama 3.2 1B end-to-end:** **425 t/s on RTX 3090**, 1.08× faster than
  llama.cpp Q8_0 (395 t/s reference). 28.9× over the v1 baseline.
- **BitNet b1.58 2B-4T end-to-end:** **214 t/s, 8/8 BOS-greedy tokens
  bit-exact** against the microsoft/BitNet llama-cli reference.
- **Kernel correctness validated** three-way for Llama 1B: scalar NumPy
  reference vs CUDA forward 16/16 bit-exact; fp32 weights (no
  ternarization) vs Q8_0 golden 7/8; ternarized vs Q8_0 golden 0/16
  (confirms H1: post-training ternarization breaks non-QAT models).
- **Op-count parity** holds: 1.002× lane-MACs vs analytical fp16 baseline
  (measured by the CPU simulator on Llama 1B).
- **Memory**: 1.58 bits/weight (8× compression vs fp16, 5× vs Q8_0).

### Underlying CUDA matmul-fabric measurement (with honest attribution)

- Packed-trit kernel beats cuBLAS INT8 TC by **1.90×** on Llama 1B at
  single-token gen (M=1), per-shape wins up to 4.18× on Wdown.
  Attribution: this advantage is bandwidth-bound (one byte yields 16
  MACs; ratio 0.94 vs 0.50 MACs/byte for INT8 ≈ 1.88×), not
  ternary-arithmetic-bound. A generic INT2 packed kernel feeding `__dp4a`
  would in principle obtain similar throughput. See paper Sec. 7.3
  "What is and isn't ternary about this work" and Cuevas (2026) §5.2.
- Our own ADD-only kernel (`mm_bitnet_addonly`) confirms: eliminating
  multiplications matches but does not exceed `__dp4a`; the
  TVMAC=0 architectural advantage of ternary needs custom silicon to
  be observable, not commodity GPU.

Full results: `docs/baselines/2026-05-10-cuda-synthesis.md`; paper
`paper/ter_paper.pdf`.

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
- [x] **F12.7** Packed-trit kernels on consumer GPU:
    - v4 wide → v6 dp4a → v7 colmaj → v8 warp → v10 unroll → v11 warp4 → v13
    - Best-per-shape dispatch: **6.67 ms / 186 GMAC/s, 1.90× over cuBLAS INT8 TC at M=1**
    - Honest attribution (per Cuevas 2026 §5.2 + our own ADD-only experiment):
      this advantage is bandwidth-bound (sub-byte packing density), not
      ternary-arithmetic-bound. An INT2 packed kernel feeding `__dp4a` would
      in principle obtain the same. Ternary-arithmetic energy advantage
      requires custom silicon (FPGA/ASIC) to be observable.
- [x] **F12.8** End-to-end forward optimization (round 2): **14.7 → 425 t/s (28.9×)**
    - Fix 1: device-pointer scale eliminates 65 D2H syncs/token → 28.9 t/s
    - Fix 2: forward kernel upgrade v4 → v11 warp-coop → 200.6 t/s
    - Fix 3: cudaGraph capture (all state on-device, entire forward as one graph) → 412 t/s
    - Fix 4: fused rmsnorm+quant + silu+quant (49 launches/token saved) → 425 t/s
    - Result: **1.08× faster than llama.cpp Q8_0** on Llama 1B, RTX 3090, 1.58 bits/weight.
    - **Kernel correctness validated**: GGUF→packed converter + NumPy reference forward + CUDA `load_model_from_bin` path. CUDA matches NumPy reference **16/16 BOS-greedy tokens bit-exact**. Diverges from Llama Q8_0 golden (0/16) per H1 (ternarization breaks non-QAT models). Architecture sanity: with fp32 weights (no ternarization) reference matches Q8_0 golden 7/8.
- [x] **F12.9** BitNet 2B-4T real-weight end-to-end forward: **214 t/s, 8/8 BOS-greedy tokens match reference**
    - GGUF i2_s converter: channel-interleaved 128-block, code mapping, per-tensor scales
    - Architectural adapters: sub_norms, ReLU², NEOX RoPE, fp16 weight-tied lm_head
    - fp16 overflow bug fixed in FFN (gate²·up exceeds 65504 → inf → zero contribution)
    - **Stage 2 (fp32 residual stream)**: 6/8 → 8/8 token match (bit-exact vs microsoft llama-cli)
    - **Stage 3 (fused rmsnorm+quant)**: +7% throughput (193 → 207 t/s)
    - Stage 1 (ADD-only kernel) brings TVMAC=0 from bench to production
    - Substrate runs real ternary-trained transformer with bit-exact reference reproduction.

## Headline numbers (Llama 1B Q8_0, RTX 3090)

| Backend | ms / forward (matmul) | GMAC/s |
|---|---:|---:|
| Mac AVX2 sim (host scalar) | 25,600 | 38 |
| ter sim CUDA naive port | 26.06 | 47.5 |
| ter sim CUDA cuBLAS sgemm | 6.87 | 180 |
| **ter sim CUDA packed-trit (best dispatch)** | **6.67** | **186** |
| cuBLAS INT8 TC (production reference) | 12.67 | 98 |

Total session-cumulative speedup vs Mac AVX2 baseline: **3,840×**.
vs cuBLAS INT8 TC at M=1 gen on Llama 1B shapes: **1.90× faster** —
attributable to packed sub-byte storage density rather than ternary
arithmetic semantics (see paper Sec. 7.3 and Cuevas 2026 §5.2).

## What's proven vs what's engineering

### Engineering results defended end-to-end
- **Llama 3.2 1B**: 425 t/s, 1.08× over llama.cpp Q8_0 on RTX 3090.
  Kernel correctness validated bit-exact vs NumPy reference (16/16).
- **BitNet b1.58 2B-4T**: 214 t/s, 8/8 BOS-greedy tokens bit-exact vs
  microsoft/BitNet llama-cli — this is where substrate-data alignment
  is semantically meaningful (QAT-trained ternary model).

### Architecturally proven (substrate-level)
- Op count parity 1.002× lane-MACs vs fp16 (CPU simulator, analytical).
- Memory compression: 1.58 bits/weight; 8× vs fp16, 5× vs Q8_0.
- Format A trit-budget tradeoff: 1+5+5 preserves Llama 1B argmax with
  logit RMSE 0.05.
- BitNet substrate-data alignment: TVMAC=0 in the matmul fabric
  (analytical + real GGUF forward verified bit-exact).

### What this paper does NOT claim (honest acknowledgments)
- **Throughput superiority of ternary arithmetic on binary silicon.**
  The 1.90× matmul win is a sub-byte packing density result; a generic
  INT2 kernel would in principle match it. Our own ADD-only kernel
  empirically matches `__dp4a` but does not exceed it.
- **Production wins at prefill regimes (M ≥ 16)** — tensor cores
  saturate; hybrid dispatch margin shrinks to ~1.04× at M=64.
- **Quality under post-training ternarization** — 0/16 vs Llama Q8_0
  golden confirms H1. Substrate-data alignment is genuine only for
  QAT-trained models (BitNet).
- **Per-op energy advantage in deployed silicon** — literature
  (Horowitz 2014, CUTIE 2020, Cuevas 2026) supports the bound, but we
  did not measure it. FPGA/ASIC prototype is the next experiment.
- **Novelty of the LUT/packed-low-bit principle** — T-MAC (Wei et al.
  EuroSys 2025), LUTMUL (Xie et al. ASPDAC 2025), Platinum (Shan et
  al. 2025), TeLLMe (Qiao et al. 2025) all precede this work. Our
  contribution is the specific GPU `__dp4a` packed-trit kernel design
  + end-to-end engine integration + bit-exact validation.

### Acknowledgment
Felipe Cuevas Araneda's *Análisis de Eficiencia Energética en
Multiplicación de Matrices con Pesos Discretizados* (mayo 2026)
provides the formal energy-cost framework (Prop. 2: ρ₃ ≥ 3.8× bound
in 45nm; Prop. 5: k=3 efficiency optimum; Prop. 6: MAC-to-LUT
reduction) and a critical review that motivated the honest framing
adopted in this revision.

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
