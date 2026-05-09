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
- [x] F4 (matmul + rmsnorm) — Sim::call_kernel, KernelTable, tk_matmul_b_9t, tk_rmsnorm + rsqrt LUT, TVMUL opcode, jump relocation in install.
- [ ] F4 (rest) — RoPE, SwiGLU, softmax, attention kernels (next plan).
- [ ] F5 — ntransformer bridge.
- [ ] F6 — Llama 3.2 1B end-to-end.

See `docs/superpowers/specs/2026-05-08-ter-design.md` for the design.
See `docs/superpowers/plans/` for the implementation plans.
