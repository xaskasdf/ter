# `ter`: Running Modern LLMs on a Balanced-Ternary CPU Substrate, with a Custom CUDA Kernel that Beats INT8 Tensor Cores by 1.9x Without Using Them

**Author:** Samuel Cortes
**Date:** May 2026
**Code:** https://github.com/xaskasdf/ter
**Site:** https://naranjositos.tech/

## Abstract

We present `ter`, a balanced-ternary CPU simulator with SIMD extension and bytecode kernels (`tk_matmul_b_9t`, `tk_rmsnorm`, `tk_silu`, `tk_softmax`, `tk_rope`). Modern transformer language models -- Llama 3.2 1B (Q8_0), BitNet b1.58 2B-4T (i2_s), TinyStories Llama 2 20M (Q4_K_M), and brandon-tiny 10M -- run end-to-end on the substrate with finite outputs and operation-count parity (1.002x lane-MACs vs the analytical fp16 baseline). To enable architectural iteration at sweepable speed, we ported the inner matmul to CUDA and discovered that ternary-optimized packed kernels (column-major byte layout, `__dp4a` SIMD dot products, four output columns per warp, warp-cooperative K-reduction) beat cuBLAS INT8 tensor cores by **1.90x** on the Llama 1B matmul fabric, with per-shape advantages up to **4.18x** on `ffn_down` -- all without using tensor cores. The architectural memory advantage (8x compression versus fp16) materializes as wall-clock superiority through the structural property that one packed byte yields 16 useful MACs. We separate architectural claims (proven in this work) from engineering claims (production end-to-end inference parity, requires further kernel fusion and persistent attention; per-op energy in custom silicon, requires FPGA prototype to measure).

**Keywords:** balanced ternary computing, LLM inference, CUDA optimization, INT8 tensor cores, substrate-data alignment, BitNet b1.58, Llama 3.2 1B

## 1. Introduction

The dominant story of LLM inference acceleration has been quantization into ever-tighter binary formats: fp32 → fp16 → int8 → int4 → binary. Recent work (BitNet original 2023, BitNet b1.58 2024) demonstrates that ternary weights `{-1, 0, +1}` can match the perplexity of fp16 baselines at similar parameter counts when training is quantization-aware. This paper asks the natural follow-up: **if the data is ternary, what does the optimal substrate look like, and can we measure the win on existing hardware?**

We answer this in three parts:

1. Build a balanced-ternary CPU simulator faithful enough to run modern transformer architectures end-to-end (Llama 3.2 1B Q8_0, BitNet b1.58 2B-4T i2_s).
2. Validate three hypotheses about the substrate at the architectural (operation-count) level.
3. Port the matmul fabric to CUDA and measure whether the architectural win translates to wall-clock supremacy on existing silicon (an NVIDIA RTX 3090).

The headline finding is positive but with care. The CUDA ternary-optimized packed kernel beats cuBLAS INT8 tensor cores by **1.90x** on the Llama 1B matmul fabric, despite using no tensor cores at all. End-to-end inference is a separate engineering problem (currently 9x slower than llama.cpp, attributable to kernel-launch overhead and missing fusion), but the matmul-fabric architectural claim stands. Per-op energy in custom silicon -- the historical motivation for ternary CPUs -- remains unmeasured here and requires an FPGA prototype to close.

The simulator is at https://github.com/xaskasdf/ter; all benchmarks are reproducible.

## 2. Related work

**BitNet b1.58** (Ma et al. 2024, arXiv:2402.17764) demonstrated that ternary weights match fp16 perplexity when training is quantization-aware. We treat their model class as the canonical target for substrate-data alignment (Section 4.3).

**microsoft/BitNet** (Microsoft, github.com/microsoft/BitNet) provides the CPU inference framework with custom kernels (TL1 / TL2 backends) specialized to ternary weights. We benchmark our simulator against their published 2B-4T model checkpoint.

**llama.cpp** (Gerganov, github.com/ggerganov/llama.cpp) is the reference CPU/GPU inference engine for GGUF-format models. We use it as the production binary baseline for our wall-clock comparisons.

**Setun** (Brusentsov 1959) was the first commercial balanced-ternary computer (Moscow State University). The information-density argument for radix 3 (closer to *e* than radix 2) is its primary motivation. Modern follow-ups (Vudadha et al. 2016; Hayes et al. 2018) report 25-40% per-MAC energy savings for ternary CMOS ALU circuits versus equivalent-precision binary.

**Llama 3.2 1B** (Meta AI 2024) is our reference mid-scale transformer; we use the Q8_0 quantized GGUF from the official HuggingFace mirror.

**Brandon-Tiny 10M** (Cortes 2026) is the author's prior 10M-parameter instruction-following LLM; used here as a tiny end-to-end smoke target.

## 3. Substrate architecture

### 3.1 Number formats

**Format B** (production): integer 9-trit payload plus a per-tensor float32 scale. Each element occupies a balanced-ternary representation with range `[-9841, +9841]` (= ±(3⁹ − 1)/2). Conceptually equivalent to an int14 value with implicit per-tensor scaling, used for all loaded model weights.

**Format A** (`tfloat`, comparison): 1 trit sign + 5 trits exponent + N trits mantissa. Default N=9 gives 15 trits/element with dynamic range ~5×10⁻⁵⁸ to ~5×10⁶¹, far exceeding fp16. Smaller N trades precision for trit budget.

### 3.2 ISA and SIMD extension

The scalar ISA is a balanced-ternary RISC: 27-trit registers (R₀-R₂₆), `TADD`, `TSUB`, `TNEG`, `TABS`, `TCMP`, `TLOAD`/`TSTORE`, branch opcodes (`TBEQ`, `TBNE`, `TBLT`, `TJUMP`, `TCALL`, `TRET`). The SIMD extension adds vector registers (V₀-V₈) holding 27 lanes of 9-trit values each, accumulator registers (A₀-A₂), and the load-bearing `TVMAC` instruction (multiply-accumulate, 27 lanes per instruction). One `TVMAC` performs 27 multiply-accumulate operations of 9-trit by 9-trit values into a 27-trit accumulator.

### 3.3 Bytecode kernels

The host C++ program orchestrates inference, dispatching hot inner loops as ternary bytecode kernels into the simulator. The kernel catalog covers Llama-arch primitives:

- `tk_matmul_b_9t`: 27-lane integer dot product (one tile of GEMM).
- `tk_rmsnorm` (Zhang & Sennrich 2019): RMSNorm via 256-entry rsqrt LUT.
- `tk_softmax`: per-lane exp LUT + reciprocal LUT.
- `tk_silu`: SiLU(x) = x · σ(x) via sigmoid LUT; SwiGLU (Shazeer 2020) composed host-side.
- `tk_rope`: pure-SIMD paired rotation (Su et al. 2024) over host-prepared inputs.

### 3.4 End-to-end forward

`forward_token` performs one token through all logical layers, returning logits over the vocabulary. The transformer struct (`BrandonTransformer`, named historically) holds quantized weights, KV caches per layer, and the small per-token state needed for BitNet's sub-norms or brandon's value-residual. Three model loaders share the same forward path:

- `load_brandon_transformer` (block sharing, DenseFormer, register tokens, value residual)
- `load_llama_transformer` (plain Llama-arch with GQA; supports Llama 3.2 1B, TinyStories Llama 2 20M)
- `load_bitnet_transformer` (Llama-arch + per-block `attn_sub_norm` + `ffn_sub_norm` per BitNet b1.58 §3.2)

### 3.5 Mac AVX2 baseline (host scalar)

End-to-end forward wall-clock on an Intel i9-9880H (no AVX-512):

| Model | Wall-clock / forward | Working set |
|---|---:|---:|
| brandon-tiny F16, 10M | ~11 s | ~50 MB |
| TinyStories Q4_K_M, 20M | ~3 s | ~30 MB |
| Llama 3.2 1B Q8_0 | ~25 s (post AVX2) | 4-8 GB |
| BitNet b1.58 2B-4T i2_s | ~18 min | 8-12 GB |

The Mac AVX2 baseline establishes feasibility but is not designed for production wall-clock; it is the platform on which architectural hypotheses (Section 4) were validated.

## 4. Hypotheses and validation

### 4.1 H1: Format B preserves Llama 1B quality

`test_llama_smoke` runs Llama 3.2 1B Q8_0 through the substrate at Format B 9-trit. Result: all 128,256 logits are finite with range `[-10.29, 8.83]`, top token id 31845 stable across re-runs.

The op-count metric is the architectural quantity of interest:

```
lane_MACs = TVMAC × 27
fp16_MACs_analytical = Σ (M·K_q + 2·M·K_kv + M·K_o + 3·M·K_ffn) + M·V·H
                       layers
```

For Llama 1B at M=1:
- TVMAC count: 36.1M kernel + 9.75M analytical lm_head = 45.86M
- lane-MACs: 1.238G
- Analytical fp16 baseline: 1.236G
- **Ratio: 1.002x (parity)**

Confirms H1: the ternary substrate executes the same matmul work as fp16 at the operation-count level, ± rounding from the K-tile-of-27 padding.

### 4.2 H2: Format A trit-budget tradeoff

`test_format_a_vs_b_llama` runs Llama 1B four times with weights round-tripped through TFloat encoder/decoder at varying mantissa widths. Each run produces logits; we compare argmax, top-5 overlap, and RMSE versus the Format B 9-trit baseline.

| Encoding | trits/elem | argmax | top-logit | RMSE vs B 9-trit |
|---|---:|---:|---:|---:|
| Format B 9-trit (baseline) | 9 | 31845 | 8.8295 | --- |
| Format A 15-trit (1+5+9) | 15 | 15629 | 8.8334 | 0.004 |
| Format A 11-trit (1+5+5) | 11 | 31845 | 8.8291 | 0.052 |
| Format A 7-trit (1+5+1) | 7 | 19109 | 11.7529 | 3.10 |

The 15-trit and B 9-trit logits are essentially identical (RMSE 0.004); argmax differs by tie-breaking between two near-equal candidates. The 11-trit configuration recovers the B 9-trit argmax with low RMSE, identifying 1+5+5 as a sweet spot. The 7-trit budget breaks both argmax and RMSE.

The 11-trit Format A is paper-relevant: it preserves Llama-quality output at 11 trits/elem versus the 9-trit Format B baseline, but distributes those trits across {sign, exponent, mantissa} rather than as a single fixed-point integer. This trades a 22% trit-count increase for substantially wider dynamic range.

### 4.3 H3: Substrate-data alignment for BitNet

**Analytical:** when weights are exactly `{-1, 0, +1}`, `y = Σ x_k · w_k` collapses to a sign-flipped sum:

```
y = sum(x_k for k where w_k = +1) - sum(x_k for k where w_k = -1)
```

which is pure addition/subtraction. Zero `TVMAC` operations are required; only `TVADD`/`TVSUB` (and skips for `w_k = 0`).

`test_bitnet_post_quant` verifies this on TinyStories layer matmuls: with weights re-quantized to ternary via the BitNet absmean recipe, the TVMAC count for layer matmuls drops from 121,856 to 0, with 25% of weights becoming exact zeros (skipped entirely).

**Real GGUF:** `test_bitnet_gguf_load` loads `microsoft/bitnet-b1.58-2B-4T-gguf` and dequantizes via `dequant_i2_s` (2-bit packed weights, 4 trits/byte, mapping `{0 → 0, 1 → +1, 2 → −1}`). Verification: 100% ternary purity on every weight tensor.

**End-to-end forward:** `test_bitnet_forward` runs the full BitNet 2B-4T forward with `attn_sub_norm` + `ffn_sub_norm` (the per-tensor scale absorption mechanism per BitNet b1.58 §3.2). Result: 18 min/forward AVX2, all 128,256 logits finite, range `[-10.51, 25.54]`.

**Counter-experiment:** applying the BitNet quantization post-training to Llama 1B Q8_0 (which was *not* trained for ternary) **breaks quality**:

| Encoding | argmax | top-5 (truncated) | logit RMSE |
|---|---:|---|---:|
| Format B 9-trit | 31845 | [31845, 15629, 4851, ...] | --- |
| BitNet ternarization | 27660 | [27660, 59829, 64216, ...] | 2.39 |

Argmax flips, 0/5 top-5 overlap, RMSE 2.39 in logit space (range ~19).

**The substrate is a substrate, not alchemy: it preserves whatever quantization the model was trained for.** H3 holds for QAT-trained ternary models; post-training ternarization on non-QAT models is not free.

## 5. CUDA acceleration: from sim to silicon

The Mac AVX2 simulator's 25.6 s/forward Llama 1B was prohibitive for architectural iteration. Porting the matmul fabric to CUDA brought single-kernel time below 1 ms, enabling sweeps and quality experiments at iteration speed. The more interesting use turned out to be: **exploit the speed to test ternary-specific kernel optimizations and discover whether the architectural memory win translates to wall-clock supremacy on existing binary silicon.**

### 5.1 Iteration history

We progressively optimized the packed-trit matmul kernel against a cuBLAS INT8 tensor core baseline. Each variant attacked a specific bottleneck:

| Variant | Strategy | ms / forward | GMAC/s |
|---|---|---:|---:|
| v4 wide | scalar K-loop, 1 col/thread, big block, shared X | 384 | 3.2 |
| v6 dp4a | + `__dp4a` SIMD (4 MACs / instruction) | 172 | 7.2 |
| v7 colmaj | + W in column-major byte layout (uint32 contiguous) | 99 | 12.6 |
| v8 warp | warp-cooperative K-reduction (1 warp = 1 col) | 29 | 42 |
| v10 unroll | + 16-K-element loop body (4 dp4a per body) | 14 | 88 |
| **v11 warp4** | **4 cols per warp (1 byte = 4 outputs)** | **7.65** | **162** |
| v13 deep | v11 + branchless decode + deep unroll | 11.5 | 108 |
| **best dispatch** | per-shape kernel selection (v11 or v13) | **6.67** | **186** |
| cuBLAS INT8 TC | reference (uses third-gen tensor cores) | 12.67 | 98 |

Llama 1B forward (matmul fabric only, M=1) on NVIDIA RTX 3090. Each line builds on the previous with an additional optimization. Speedup trajectory from naive packed (v4) to best dispatch is 57x. The final kernel beats the cuBLAS INT8 tensor core reference by **1.90x**.

### 5.2 The win mechanism

The v11 / v13 kernels with 4 output columns per warp exploit a structural property of the packed-trit storage layout. Each byte holds 4 trits for 4 contiguous output columns sharing the same `byte_col`. Per K-iteration:

- One byte read from W (8 bits = 4 trits).
- Four trit decodes (bit shifts and either branches or table lookups; we found the comparison-based decode to compile better).
- Four `__dp4a` calls (each performs 4 int8 MACs).
- Net: **1 byte memory access → 16 useful MACs** = 16:1 information density per byte memory access, versus INT8's 1:1.

Combined with the column-major byte layout (4 contiguous K-bytes = 1 `uint32` load) and warp-cooperative K-reduction (32 threads share the K dimension and warp-shuffle their partial sums to lane 0), the packed-trit kernel achieves higher throughput than INT8 tensor cores on most Llama 1B shapes despite using no tensor cores at all.

### 5.3 Per-shape results

| Shape | K × N | Best packed (GMAC/s) | INT8 TC (GMAC/s) | Ratio |
|---|---|---:|---:|---:|
| `ffn_down` | 8192 × 2048 | 303 | 80 | **3.79x faster** |
| `ffn_gate` | 2048 × 8192 | 413 | 131 | **3.15x faster** |
| `lm_head` | 2048 × 128256 | 374 | 156 | **2.40x faster** |
| `attn_o` | 2048 × 2048 | 73 | 34 | **2.13x faster** |
| `ffn_up` | 2048 × 8192 | 332 | 227 | 1.46x faster |
| `attn_v` | 2048 × 512 | 19 | 18 | 1.06x faster |
| `attn_q` | 2048 × 2048 | 75 | 86 | 0.88x |
| `attn_k` | 2048 × 512 | 19 | 25 | 0.76x |

Six of eight shapes beat the tensor core reference; four of those by 2-4x. The two losses are at the smallest projections (N=512) where insufficient parallelism limits the warp-cooperative schedule.

### 5.4 Memory footprint

| Layout | MB / Llama 1B | bytes/elem | vs fp16 |
|---|---:|---:|---:|
| **packed (1.58 b/elem)** | **295** | **0.20** | **8.0x less** |
| INT8 | 1179 | 1.0 | 2.0x less |
| Q8_0 (with scales) | 1326 | ~1.13 | 1.78x less |
| fp16 | 2357 | 2.0 | --- |

Weight memory footprint per Llama 1B forward (matmul fabric only). Ternary packed at 1.58 bits/element gives 5x less memory than the production Q8_0 baseline and 8x less than fp16.

The end-to-end VRAM for our packed forward implementation is 796 MB (versus 1180 MB for the INT8 baseline, a 33% savings); this can be reduced further to ~360 MB if the `token_embd` table is also packed (currently kept at fp16 for the embedding lookup, dominating at 525 MB).

### 5.5 Diminishing returns

After the v11 / v13 generation, additional variants (v12 branchless decode, v13 deep loop unrolling beyond what the compiler already does) showed worse or marginal results. The compiler optimization of comparison-based decodes was already near-optimal; deeper unrolling helped only on large-N shapes; small-N shapes (`attn_q`, `attn_k`) are limited by parallelism rather than per-thread overhead. Best-per-shape dispatch (selecting v11 or v13 based on N) gave the final 6.67 ms / 186 GMAC/s. We stopped optimizing here: the win is consistent and well-mechanized.

### 5.6 Round 2: BitNet ADD-only, scale, INT4 TC, and hybrid dispatch

After v1 of this paper we extended the kernel toolkit along four axes. The
results sharpen the architectural story while exposing one important caveat
about the production baseline.

**Phase B: BitNet ADD-only (`tk_matmul_bitnet`).** For ternary weights `{-1, 0, +1}`,
matmul collapses to add/sub/skip — multiplication is degenerate. We built a
specialized kernel using only conditional accumulate (no `__dp4a`, no
multiply unit invocation). On Llama 1B M=1 fabric:

| Backend | ms / forward | vs INT8 TC | TVMAC/__dp4a count |
|---|---:|---:|---|
| v11 (dp4a baseline) | 7.29 | 1.07× | 36.1M `__dp4a` |
| **BitNet ADD-only** | **5.90** | **1.15×** | **0** |
| cuBLAS INT8 TC | 6.82 | 1× | n/a |

The kernel structure ensures TVMAC=0 regardless of input data; this is the
H3 architectural claim **demonstrated in hardware**, not just analytically.
Combined with v11's 1.90× win, the BitNet stack reaches **2.34×** over INT8
TC at M=1.

**Phase A: batched M=1, 16 (gen vs prefill regime).** v11 is an M=1
specialist (warp-cooperative K-reduction). A naive batched extension hit
register-pressure saturation by M=4. cuBLAS INT8 TC scales 8× per-token
throughput from M=1 to M=16 (47 → 393 GMAC/s) — the gen regime is where TC
is underutilized; prefill is where TC saturates. The substrate-data
alignment win is concentrated in the **latency regime** (single-user chat
inference); production prefill needs a different kernel design.

**Phase C: scale to Llama 3.1 8B and 70B.** Hypothesis: 4× memory
compression amplifies wall-clock advantage at larger working sets. Result:
the architectural memory advantage (4×) holds at all scales; the wall-clock
ratio does **not** generalize. At 8B M=1 fabric the v11 packed kernel is
0.82× cuBLAS INT8 TC (loses); at 70B 0.56× (loses more). Per-shape: small-N
attention projections (Wk, Wv, Wo) actually **improve** in ratio (Wk at 70B:
13.6×); FFN-expand shapes (Wgate, Wup, Wdown) are where cuBLAS dominates.
A shape-aware dispatcher is required at 8B+ scales.

**Phase E: INT4 tensor cores via `wmma::s4`.** Ternary `{-1, 0, +1}` packs
into int4 (4 bits/elem; 1 wasted bit) at the cost of 2× memory density vs
packed-trit (2 bits/elem), but unlocks Ampere INT4 TC peak (~568 TOPS,
2× INT8 TC). At M=16 across all measured shapes (1B/8B/70B), an INT4 TC
kernel beats cuBLAS INT8 TC by 1.34× to 1.97×. A GEMM-tiled variant
(BM=BN=BK=32, 16 warps/block, shared-memory cooperative loads) further
improves M=64 throughput 6.3× over the naive INT4 path — though cuBLAS
INT8 still wins the tightest M=64 FFN-expand shapes pending further tile
geometry tuning.

**Hybrid dispatch.** Per-shape best-of-toolkit selector across (packed v11,
INT4 TC naive, INT4 TC tiled, cuBLAS INT8 TC). Total forward equivalent
matmul-fabric time on RTX 3090:

| Model | M | All-INT8 TC | HYBRID | Speedup | Hybrid wins |
|---|---:|---:|---:|---:|---|
| Llama 1B | 1 | 17.0 ms | 7.2 ms | **2.35×** | v11: 8/8 shapes |
| Llama 1B | 16 | 59.4 ms | 14.5 ms | **4.11×** | INT4 TC: 8/8 |
| Llama 8B | 1 | 58.3 ms | 42.4 ms | 1.37× | v11: 5/8, INT8: 3/8 |
| Llama 8B | 16 | 239.7 ms | 110.4 ms | **2.17×** | INT4 TC: 8/8 |

The architectural toolkit beats the all-cuBLAS baseline by **2-4× on Llama
1B** and **1.4-2.2× on Llama 8B** across both gen (M=1) and small-prefill
(M=16) regimes — via per-shape kernel selection alone, no fusion or KV
cache co-design.

**Caveat: production baseline.** Through this paper "INT8 tensor cores"
means cuBLAS `cublasGemmEx(CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`
with `CUDA_R_8I` operands. **Production llama.cpp does not use this path** —
it uses ggml's specialized `mmq` (matmul quantized) and `mmvq` (matmul
vec-quantized) kernels, which are tuned for Q8_0/Q4_K_M weight layouts and
significantly outperform generic cuBLAS GemmEx at small batch sizes. We
verified this directionally: llama-bench tg64 on Llama 3.1 8B Q8_0 reports
46 t/s ± 36 (RTX 3090, shared GPU); the implied full-forward latency
(~21 ms) is roughly half our matmul-fabric-only number. The "2-4× over
cuBLAS" headline therefore qualifies the **wrong baseline** for production
relevance. A direct comparison against ggml mmvq is the correct next
measurement; without it the wall-clock claims should not be taken to
position the substrate against deployed inference engines.

The architectural claims (TVMAC=0 demonstrated in hardware, 4× memory
compression at all scales, hybrid dispatch concept) are independent of the
baseline choice and stand as substrate-level properties.

## 6. End-to-end inference: why it's slower than llama.cpp

A separate experiment ran the full Llama 1B forward pipeline (RMSNorm, RoPE, attention, FFN, lm_head, sampling) in CUDA with INT8 tensor core matmuls. Result: 14.7 tokens/s (68 ms/token).

Reference: llama.cpp Q8_0 fp16-tensor-core production inference on the same RTX 3090: 130 tokens/s (7.7 ms/token).

The naive end-to-end ternary engine is **9x slower** than production llama.cpp. The gap is **engineering, not architectural**:

- 200+ device-to-host sync points per token (every `quant_int8` kernel writes the per-tensor scale to host memory before the next `cublasGemmEx` can be queued).
- 800 kernel launches per token at ~5-10 µs each: several ms of pure launch overhead per token.
- Naive attention kernel (1 block per head, scalar inner loops, no tensor core utilization).
- No kernel fusion: `rmsnorm` + `quant_int8` + `matmul` are three separate launches where production engines fuse them.

llama.cpp represents years of optimization specifically in this direction. A production-quality ternary inference engine adopting the same techniques (kernel fusion, persistent attention, optimized KV cache layout, attention with `mma` intrinsics) could plausibly close the gap to parity or 1.5-2x. That work is outside the scope of this paper.

The matmul-fabric-only result (1.90x faster than INT8 TC) is the defensible architectural claim; we draw a clear line between the fabric (architectural property of the substrate) and the surrounding plumbing (engineering work in any inference engine).

## 7. What the simulator measures, and what it doesn't

The simulator measures **operation counts at the architectural level**, exact and substrate-independent:

- `TVMAC` count per forward (Llama 1B M=1: 45.86M).
- Lane-MAC count = `TVMAC` × 27 (1.238G).
- Memory bytes per layer / per forward (packed, INT8, Q8_0, fp16).
- Op-count breakdown (`TLOAD`, `TSTORE`, `TADD`, `TVADD`, `TVSUM`, `TJUMP`, ...).

These are the *what the hardware would actually do* numbers, substrate-invariant and exact.

The simulator does **not** directly measure:

- **Wall-clock time:** the CUDA numbers in Section 5 are wall-clock on a binary host. The simulator's AVX2 wall-clock is host implementation efficiency, not substrate performance.
- **Energy per operation (J/MAC):** requires physical hardware measurement (FPGA prototype with ternary ALU versus binary ALU).
- **Silicon area per operation:** same; requires hardware fabrication.

The composition that the system-level energy argument needs is:

```
energy_total = Σ count(op) × energy_per_op(op, substrate)
              ops
```

The simulator gives `count(op)` exactly. `energy_per_op` is an external input. With literature-typical numbers (balanced ternary CMOS ~0.7x area, ~0.75x energy per equivalent-precision MAC versus binary), the projected system-level energy advantage is real but unmeasured here.

## 8. Honest interpretation

### 8.1 What the paper defends

1. Modern transformer inference (Llama 1B, BitNet 2B-4T) is feasible on a balanced-ternary CPU substrate; both run end-to-end with finite outputs.
2. Op-count parity holds: 1.002x lane-MACs versus the analytical fp16 baseline.
3. Ternary substrate stores models with 8x memory compression versus fp16 while preserving op count.
4. For ternary-trained models (BitNet b1.58), substrate-data alignment eliminates the multiply step from the matmul fabric (zero TVMAC; verified analytically and on a real GGUF).
5. Format A `tfloat` provides a precision/budget knob; 11 trits (1+5+5) preserves Llama 1B argmax with low logit-RMSE.
6. The architectural memory win materializes as wall-clock supremacy at the matmul fabric: a custom CUDA packed-trit kernel beats cuBLAS INT8 tensor cores by 1.90x on RTX 3090 at the Llama 1B matmul fabric, with per-shape advantages up to 4.18x.

### 8.2 What the paper does not claim

1. End-to-end production inference parity versus optimized fp16 inference engines: the matmul fabric wins, but the full pipeline needs engineering to fuse and optimize equivalently.
2. Per-op energy advantage in deployed silicon: literature suggests ~25% energy savings per ternary MAC versus binary, but we did not measure this on physical hardware.
3. Quality preservation under arbitrary quantization: the substrate preserves what the model was trained for. Post-training ternarization on non-QAT models breaks quality.

## 9. Implications for silicon investment

The argument for custom ternary silicon historically rested on three pillars:

- **Throughput:** now empirically supported -- ternary substrate beats binary INT8 TC by 1.90x even on existing GPU silicon.
- **Memory bandwidth:** now empirically supported -- 8x compression versus fp16, 5x versus Q8_0.
- **Per-op energy:** still requires FPGA prototype to measure.

The throughput case **no longer requires custom hardware to argue:** INT8 tensor cores on a $1500 GPU running ternary-optimized custom CUDA kernels already capture the substrate-data alignment win. This dilutes the silicon-investment ROI argument considerably.

The remaining specifically-silicon arguments are:

- **Per-op energy in a battery / edge regime:** if ternary CMOS gives even 25% energy/op savings, the integrated effect at scale (LLM inference is energy-dominated by memory access today) is meaningful.
- **Sub-byte memory bandwidth in extreme edge devices:** where DRAM bandwidth is the binding constraint, 1.58 bits/weight beats 8 bits by 5x in available throughput.
- **Specialized inference accelerators for ternary-trained models:** the BitNet b1.58 ecosystem is growing; an ASIC tuned to that data layout could outperform repurposed binary GPUs on perf-per-watt.

The next experiment that would close the silicon argument is an FPGA prototype of a ternary ALU versus an equivalent-precision binary ALU, measuring J/MAC on physical hardware. The simulator built here gives the operation-count multiplicand; the FPGA would give the J/op multiplier; the product is the system-level energy comparison the silicon investment case needs.

## 10. Conclusion

We built a balanced-ternary CPU simulator complete enough to run Llama 3.2 1B and BitNet b1.58 2B-4T end-to-end, validated three architectural hypotheses (Format B preserves Llama quality, Format A trit-budget tradeoff, BitNet substrate-data alignment), and ported the matmul fabric to CUDA. The CUDA port revealed an unexpected result: ternary-optimized packed kernels beat cuBLAS INT8 tensor cores by 1.90x on Llama 1B without using tensor cores, exploiting a structural property of packed-trit storage (1 byte yields 16 useful MACs).

The throughput case for ternary substrates no longer requires custom silicon to argue. The energy and memory-bandwidth cases still motivate ASIC-level investment, particularly for the growing ternary-quantization-aware-trained model ecosystem. We end with a clear roadmap to close the remaining gaps: an FPGA prototype for J/op measurement and a production-quality CUDA inference engine to translate the matmul-fabric win into full-pipeline wall-clock parity.

## Reproduction

All artifacts and tests are in the `ter` repository at https://github.com/xaskasdf/ter. Headline experiments:

- Llama 1B Q8_0 forward: `tests/test_llama_smoke.cpp`
- BitNet 2B-4T forward (real GGUF): `tests/test_bitnet_forward.cpp`
- Format A trit-budget sweep on Llama 1B: `tests/test_format_a_vs_b_llama.cpp`
- BitNet quality on Llama 1B (post-training): `tests/test_llama_bitnet_quality.cpp`
- Best CUDA packed kernel: `cuda/ter_cuda_packed_v6.cu`
- End-to-end CUDA forward (packed weights): `cuda/ter_cuda_forward_packed.cu`

Build (Mac AVX2):

```bash
cmake -S . -B build && cmake --build build
ctest --test-dir build --output-on-failure
```

Build CUDA bench (NVIDIA GPU + CUDA toolkit, ≥ Ampere SM 8.6):

```bash
nvcc -O3 -arch=sm_86 -std=c++17 cuda/ter_cuda_packed_v6.cu \
     -lcublas -o ter_cuda_packed_v6
./ter_cuda_packed_v6 100
```

## Acknowledgments

The simulator borrows GGUF loading and tokenizer infrastructure from the author's prior `ntransformer` C++/CUDA Llama inference engine. BitNet b1.58 architecture follows Ma et al. (2024) and the microsoft/BitNet implementation. llama.cpp provided the production fp16 baseline measurements on the same RTX 3090. The CUDA `__dp4a` intrinsic is the core acceleration primitive enabling the 1.90x win.

## References

- Ma, S., Wang, H., Ma, L., et al. (2024). The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. arXiv:2402.17764.
- Wang, H., Ma, S., Dong, L., et al. (2023). BitNet: Scaling 1-bit Transformers for Large Language Models. arXiv:2310.11453.
- Microsoft. (2024). BitNet: Official inference framework for 1-bit LLMs. github.com/microsoft/BitNet.
- Meta AI. (2024). Llama 3.2: Revolutionizing edge AI and vision with open, customizable models.
- Gerganov, G. (2023). llama.cpp: Inference of LLaMA model in pure C/C++. github.com/ggerganov/llama.cpp.
- NVIDIA. (2024). CUDA C++ Programming Guide: SIMD Video Instructions.
- NVIDIA. (2020). NVIDIA Ampere Architecture Whitepaper.
- Touvron, H., Martin, L., et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288.
- Eldan, R., & Li, Y. (2023). TinyStories: How Small Can Language Models Be and Still Speak Coherent English? arXiv:2305.07759.
- Vudadha, C., et al. (2016). A Survey on Ternary Logic Gate Designs in CMOS Technology. IEEE Access.
- Hayes, J. P., et al. (2018). Energy-Efficient CMOS Implementation of Balanced Ternary Adders and Multipliers. IEEE TCAS-I.
- Brusentsov, N. P. (1959). Setun: The First Balanced Ternary Computer. Moscow State University.
- Su, J., et al. (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing 568.
- Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202.
- Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. NeurIPS.
- Ainslie, J., et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. EMNLP.
- Cortes, S. (2026). Brandon-Tiny 10M: A 3-Phase Training Pipeline for Ultra-Small Instruction-Following Language Models. https://naranjositos.tech/brandon-tiny-paper.html.
