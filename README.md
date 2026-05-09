# ter

Balanced ternary CPU simulator + SIMD extension. Phase F0-F2 substrate
for running Llama 3.2 1B forward-pass kernels (later phases).

## Build

    cmake -S . -B build
    cmake --build build
    ctest --test-dir build --output-on-failure

The matmul phase-gate test requires numpy. A local venv is provided at
`.venv/` (use `uv venv .venv && uv pip install --python .venv/bin/python numpy`
to recreate it).

## Status
- [x] F0 — Trit, Tryte, Word27, Word54 primitives, packing, Memory.
- [x] F1 — Scalar ISA, assembler, simulator, sum(1..5) smoke test.
- [x] F2 — SIMD extension (tvadd, tvmac, tvsum, ...), 64x64 matmul gate.
- [x] F3 — Format B quantizer (bf16/float ↔ fixed-point trit + per-tensor scale).
- [x] F4 — All kernels (matmul, rmsnorm, softmax, silu, rope) + attention via host-orchestrated composition; complete K3 transformer building blocks.
- [x] F5.1 — vendor/ntransformer/ infra lifted (Tensor, types, config, loader, tokenizer, sampler), CUDA-stripped, smoke-tested.
- [x] F5.2 — Single-layer forward (`ter::tx::forward_layer()`) via matmul kernel matches numpy (max_rel ~0.001 on tiny shapes).
- [x] F5.3a — Multi-token attention with KV cache; sequential forward_layer over 4 positions matches numpy.
- [x] F5.3b — All 4 transcendental kernels plumbed into forward_layer. Every arithmetic op now happens inside a kernel.
- [x] F5.4a — Load brandon-tiny f16 GGUF + quantize one tensor (token_embd, MSE 2.4e-10) through Format B.
- [x] F5.4b — GGUF parser recognises all 15 `brandon.*` keys; BrandonConfig populates correctly (block_count=12, compute_layer_count=24, n_registers=4, layer_map[24], use_dwa, use_value_residual, weight_tying).
- [x] F5.4c — Brandon SPM tokenizer working: vocab=8192, ChatML `<|im_start|>=4 <|im_end|>=5` resolved by string search, encode/decode round-trip on "hello world" returns 3 tokens. Auto-detect SPM-vs-GPT2 worked.
- [ ] F5.4d — Tile-aware rmsnorm/softmax for hidden_size > 27 (brandon dim=256 needs ~10 tiles).
- [ ] F5.4e — Multi-layer Transformer + brandon-specific bits (value_residual, DWA, register prefill); first generated token.
- [ ] F5.4f — TinyStories Q4_K_M (test the unpacker for tinier validation runs).
- [ ] F6 — Llama 3.2 1B Q8_0 end-to-end (paper target).

## Building blocks (F0-F4 complete)

| Component | Purpose | File |
|---|---|---|
| `tk_matmul_b_9t` | 27-length integer dot product (one tile of GEMM) | `src/kernels/tk_matmul_b_9t.tasm` |
| `tk_rmsnorm` | RMSNorm via rsqrt LUT | `src/kernels/tk_rmsnorm.tasm` |
| `tk_softmax` | Softmax via per-lane exp LUT + recip | `src/kernels/tk_softmax.tasm` |
| `tk_silu` | SiLU(x) = x · sigmoid(x); SwiGLU via host composition | `src/kernels/tk_silu.tasm` |
| `tk_rope` | Pure-SIMD paired rotation (host-prepared inputs) | `src/kernels/tk_rope.tasm` |
| Attention | Single-head Q/K/V + RoPE + scores + softmax + out, via host composition | `tests/test_attention.cpp` |

See `docs/superpowers/specs/2026-05-08-ter-design.md` for the design.
See `docs/superpowers/plans/` for the implementation plans.
