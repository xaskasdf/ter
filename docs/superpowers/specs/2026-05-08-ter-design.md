# `ter` — Ternary CPU Simulator + Llama 3.2 1B Forward Pass

**Status:** Approved design (brainstorm complete) — pending implementation plan
**Date:** 2026-05-08
**Author:** Samuel Cortes Rojas (`samuel.cortes.rojas@gmail.com`)
**Repo:** `~/ter`

---

## 1. Vision

A C++ simulator of a **balanced ternary CPU with a SIMD vector extension**, capable of executing the forward pass of `meta-llama/Llama-3.2-1B` implemented as **ternary bytecode kernels** invoked from a host orchestrator that reuses the loader, tokenizer, and sampler from `~/ntransformer`.

The project's purpose is to test a thesis: that a transformer running on ternary hardware (with weights expressed in ternary fixed-point) reduces arithmetic operation counts vs an equivalent binary fp16 path, while preserving output quality. The simulator runs on a binary host, so no wall-clock speedup is expected; the result is measured in **operation counts**, not seconds.

### Target audience for results

A short paper / write-up demonstrating the thesis with reproducible measurements. Hardware fabrication is out of scope.

---

## 2. Locked Decisions (from brainstorm)

| Axis | Decision |
|---|---|
| Number system | **Balanced ternary** `{-1, 0, +1}` (Setun lineage) |
| Reference architecture | **Llama 3.x** (RMSNorm + RoPE + SwiGLU + GQA) |
| Validation model | **Llama 3.2 1B** (`meta-llama/Llama-3.2-1B`) |
| Stretch model | TinyLlama / GPT-2 / larger Llama 3 (memory-permitting) |
| Substrate | **Software simulator** on a binary host (no real ternary HW) |
| Number representation, MVP | **Format B**: fixed-point ternary integer + per-tensor `float32` scale |
| Number representation, phase 2 | **Format A**: native ternary float (`tfloat`) — comparison and quality study |
| ISA shape, MVP | **B**: scalar RISC core + ternary SIMD extension (`tvmac`-centric) |
| ISA shape, late stage | **A**: pure-scalar RISC for honest TADD counting (documented, optional) |
| Implementation language | **C++** for sim core and host; **Python** for tooling (GGUF, quantizer, dumps) |
| Kernel topology | **K3 hybrid**: host C++ orchestrates, hot kernels (matmul, RMSNorm, RoPE, SwiGLU, softmax, attention) execute as **ternary bytecode** on the sim |
| Migration path | K3 → **K2** (sim-resident orchestrator, full ternary program) → **K4** (bare-metal x86 deployment as ring-0 kernel; sim hyper-optimised against binary substrate) |
| Reuse strategy | Lift `loader`, `tokenizer`, `sampler`, `config`, `transformer.cpp` orchestration from `~/ntransformer`; replace CUDA backend with a `TernarySim` backend |
| Bare-metal target (K4) | Integrate as a backend into `~/osito-k`'s in-kernel LLM inference path, replacing or complementing the AVX2/FMA tensor kernels |

---

## 3. Repository Layout

```
ter/
├── README.md
├── CMakeLists.txt
├── .gitignore
├── docs/
│   ├── superpowers/specs/             # design artifacts from brainstorm
│   ├── isa.md                         # full ISA reference (opcodes, encoding)
│   ├── number-formats.md              # format B (fixed-point) + format A (tfloat)
│   ├── memory-map.md                  # sim memory regions, alignment, addressing
│   └── kernel-bytecode.md             # K3 calling convention, ABI, kernel catalog
├── include/ter/
│   ├── trit.hpp                       # Trit, Tryte, Word27, Word54, packing helpers
│   ├── isa.hpp                        # opcode enum, instruction encoding/decoding
│   ├── sim.hpp                        # public Sim API
│   └── kernels.hpp                    # kernel ID table, invocation helpers
├── src/
│   ├── core/                          # ternary primitives, arithmetic, packing
│   ├── sim/                           # CPU sim: fetch/decode/execute, memory, regs
│   ├── asm/                           # text-to-bytecode assembler
│   ├── kernels/                       # ternary bytecode kernels (.tasm + emitted bin)
│   ├── numfmt/                        # bf16 ↔ format B ↔ format A converters
│   ├── host/                          # orchestrator (forward.cpp, sim_backend.cpp)
│   └── main.cpp                       # CLI entry point
├── vendor/ntransformer/               # lifted-and-stripped from ~/ntransformer
│   ├── core/tensor.{h,cpp}            # extended with TritFP_B dtype, Sim device
│   ├── model/{config,loader,transformer,attention,ffn,norm}.*
│   └── inference/{tokenizer,sampler}.*
│   # NOTE: cuda/* is intentionally dropped
├── tools/
│   ├── decompose_gguf.py              # from ntransformer (drop-in)
│   ├── quantize_to_trit.py            # bf16 → format B packed trits
│   └── trit_dump.py                   # sim-memory inspector (humans-readable trits)
└── tests/
    ├── test_trit_arith.cpp            # property-based, exhaustive on small ranges
    ├── test_isa_decode.cpp            # encoding round-trip, illegal-op detection
    ├── test_sim_smoke.cpp             # tiny programs (factorial, fib) in tasm
    ├── test_quantize_roundtrip.cpp    # bf16 → trit → bf16, MSE bounds
    ├── test_kernel_matmul.cpp         # matmul vs numpy reference
    └── test_e2e_tinyllama.cpp         # small Llama-arch model end-to-end
```

The `vendor/ntransformer/` tree is a lift (a fork copy), not a submodule, because we need to surgically modify the dtype layer and remove CUDA dependence. The original repo remains independent.

---

## 4. Ternary Primitives

### 4.1 Type hierarchy (all powers of 3 — clean symmetry)

| Type | Trits | Range | Primary use |
|---|---|---|---|
| `Trit` | 1 | `{-1, 0, +1}` | atom; host storage `int8_t` with values restricted to `{-1, 0, 1}` |
| `Tryte` | 3 | `[-13, +13]` | opcodes, register indices, small immediates |
| `Word27` | 27 | `[-3.8e12, +3.8e12]` | scalar register, address, full instruction word |
| `Word54` | 54 | `[-1.4e25, +1.4e25]` | accumulator (matmul without overflow) |
| `Vec` | 27 lanes × 9 trits | per lane `[-9841, +9841]` | SIMD register |

### 4.2 Host-side packing

- **Encoding:** 2 bits per trit. `00→0`, `01→+1`, `10→-1`, `11→reserved (illegal)`.
- A `Word27` packs into 54 bits → fits in a single `uint64_t` with 10 unused bits.
- Sim memory backing store: `std::vector<uint64_t>`, addressable per-word.
- Memory waste vs ideal base-3 packing: ~33%. Acceptable for MVP. A future denser packing (5 trits per byte using base-3 digits) is a drop-in replacement behind the `Memory` API.

### 4.3 Ternary arithmetic primitives (host-side reference impl)

`Trit + Trit` produces a `(Trit sum, Trit carry)` tuple. This is the lookup table the sim uses to implement `tadd` for arbitrary widths. All higher-level ternary arithmetic (Word27 add, Word54 add, multiply, negate) builds on this primitive. Because the system is balanced, **negation is per-trit inversion** — there is no two's-complement-style sign manipulation.

---

## 5. ISA Specification

### 5.1 Register file

```
R0..R26    27 scalar registers, Word27 each (R0 conventionally hard-zero)
V0..V8     9  vector registers, Vec each (27 lanes × 9 trits = 243 trits)
A0..A2     3  Word54 accumulators (target of tvmac)
PC         Word27 program counter
SR         Tryte status (sign trit of last result, halt flag, fault flag)
```

### 5.2 Instruction format

Fixed 27-trit width (one `Word27`):

```
┌──────────┬─────┬─────┬─────┬─────────────────┐
│  opcode  │ dst │ src1│ src2│   imm / ext     │
│ 6 trits  │  3  │  3  │  3  │    12 trits     │
└──────────┴─────┴─────┴─────┴─────────────────┘
   729 ops    27    27    27       ±265720
```

The opcode's most-significant trit categorises:
- `−` → system / control / IO
- `0` → scalar arithmetic / memory
- `+` → SIMD / vector

### 5.3 MVP opcode catalogue (F1–F2)

```
SCALAR        SIMD                  MEMORY      CONTROL    SYSTEM
─────────     ───────────────       ───────     ──────     ──────
tadd          tvadd                 tload       tbeq       tnop
tsub          tvsub                 tstore      tbne       thalt
tneg          tvneg                 tloadi      tblt       tdbg
tabs          tvbroadcast           tvload
tand3         tvmac (A += a*b)      tvstore     tjump
tor3          tvsum (Σ lanes)                   tcall
txor3         tvmax                             tret
tcmp          tvshuf
tsign
```

`tvmac` is the load-bearing instruction: `A_acc += V_a · V_b` over 27 lanes per single instruction. This is where the thesis lives — every transformer matmul reduces to a sequence of `tvmac`s plus per-tile scale fix-ups.

The full instruction reference, encoding tables, and worked examples live in `docs/isa.md`.

---

## 6. Sim Core

```cpp
class Sim {
  Memory       mem;          // ~4 GB target backing store
  RegFile      regs;         // R, V, A, PC, SR
  KernelTable  kernels;      // kernel_id → entry address
  OpCounters   ops;          // per-opcode-class counters

  // Public API
  void   load_program(const KernelBlob&);
  Word27 alloc(size_t trits);
  void   dma_in(const void* host, Word27 sim_addr, size_t bytes);
  void   dma_out(Word27 sim_addr, void* host, size_t bytes);
  void   call_kernel(KernelId id, std::span<const Word27> args);
  void   run();   // loops until thalt
  const  OpCounters& counters() const;
};
```

The fetch/decode/execute loop is conventional. Each opcode has a handler that mutates `regs` and/or `mem`, and bumps the appropriate `OpCounters` field. Counters are the publishable artefact — they let us report e.g. *"forward pass of layer 0 used 47.3 M tvmacs and 0 tmuls"*.

### 6.1 Memory map of the simulated machine

| Region | Purpose | Approximate size |
|---|---|---|
| `0x0000_0000 …` | Code segment (loaded kernels) | up to 16 MB |
| `0x0100_0000 …` | Weight tensors (read-only after load) | ~2.25 GB for Llama 3.2 1B |
| `0xE000_0000 …` | KV cache | up to 256 MB |
| `0xF000_0000 …` | Activations / scratch | up to 512 MB |
| `0xFF00_0000 …` | Stack | 1 MB |

Addresses above are illustrative; ternary equivalents are documented in `docs/memory-map.md`. Total host-side budget: ~4 GB of `uint64_t`-backed memory.

---

## 7. Number Formats

### 7.1 Format B — fixed-point ternary + per-tensor scale (MVP)

Per tensor, store:
- `dtype = TritFP_B`
- `n_trits_per_elem` (default 9; configurable)
- `scale: float32` (per tensor)
- `zero_point: 0` (balanced ternary needs no zero-point)
- `shape`
- payload: `n_elems × n_trits_per_elem` packed at 2 bpt

Conversion `bf16 → trit`:
```
max_trit_int = (3^n_trits_per_elem - 1) / 2
scale        = max(|tensor|) / max_trit_int
trit_int     = round(value / scale)
trits        = balanced_ternary_digits(trit_int, n_trits_per_elem)
```

Quality target with 9 trits/element: equivalent to int10 (~14.27 effective bits), which is comfortably above int8 PTQ. If empirical quality drops below the gate (see §10), the constant `n_trits_per_elem` is bumped to 12 with no other code changes.

The matmul pipeline `Y = X · Wᵀ` under format B:
```
1. Load V_x  (lane = activation tryte)
2. Load V_w  (lane = weight tryte)
3. tvmac A0, V_x, V_w        # A0 += sum(x_i * w_i), all integer ternary
4. (loop over lanes to cover the dot product)
5. acc_int = read A0
6. y_float = scale_x * scale_w * acc_int     # only float op, on host
```

**Inside the sim there are zero floating-point operations.** All math is integer ternary. Float scaling lives on the host, applied once per output tile. This is the operational embodiment of the thesis.

### 7.2 Format A — `tfloat` ternary float (phase 2, comparison)

Native ternary FP with the layout:
```
┌─────┬──────────────┬────────────┐
│  s  │   exponent   │   mantissa │
│ 1t  │   5t (±121)  │    9t      │
└─────┴──────────────┴────────────┘
```

Sign is free (one trit, balanced). Add and multiply are implemented natively in the ternary ALU. Conversion from bf16 is straightforward but lossy. The quality and op-count comparison vs Format B is a publishable result (see §10).

Detailed encoding, rounding mode, and special values are in `docs/number-formats.md`.

---

## 8. Kernel Bytecode (K3)

Each kernel is a `.tasm` source file assembled to a contiguous blob of `Word27` instructions, loaded into the code segment of sim memory. The host invokes a kernel with:

```cpp
sim.call_kernel(KernelId::matmul_b_9t, {addr_x, addr_w, addr_y, M, N, K, scale_xw});
```

This pushes the current PC, places arguments in `R1..R7`, jumps to the kernel's entry, and runs the sim until `tret`.

### 8.1 Calling convention

| Register | Role |
|---|---|
| `R0` | hard-zero |
| `R1..R7` | argument and return registers |
| `R8..R15` | caller-saved scratch |
| `R16..R25` | callee-saved |
| `R26` | stack pointer |
| `V0..V8` | vector scratch (caller-saved) |
| `A0..A2` | accumulator scratch (caller-saved) |

Arguments beyond `R7` are passed on the stack. Vector arguments are passed by sim-memory pointer.

### 8.2 MVP kernel catalogue

| Kernel | Args | Approx. asm lines | Builds on |
|---|---|---|---|
| `tk_matmul_b_9t` | `X*, W*, Y*, M, N, K, scale_xw` | ~80 | `tvmac` core loop |
| `tk_rmsnorm` | `X*, W*, Y*, N, eps_recip` | ~40 | `tvadd`, `tvsum`, `tisqrt` |
| `tk_rope` | `X*, freqs*, pos, head_dim` | ~60 | rotation pairs |
| `tk_swiglu` | `gate*, up*, Y*, N` | ~50 | `tsigmoid_lookup`, `tvmul` |
| `tk_softmax` | `logits*, Y*, N` | ~70 | `texp_lookup`, `tvsum`, `tvdiv` |
| `tk_attn_score` | `Q*, K*, S*, head_dim, n_kv` | ~120 | composition of above |

`tisqrt`, `tsigmoid_lookup`, `texp_lookup` are implemented as ternary lookup tables (256–2048 entries, memory-resident). This mirrors how real CPUs handle transcendentals (CORDIC / LUT). An "honest mode" flag forces Newton-Raphson in trits and reports separate counters, so we can quantify how much of the win comes from LUTs vs ternary itself.

### 8.3 Assembler

A small textual assembler (`src/asm/`) translates `.tasm` to `Word27` blobs. C++ macros allow emitting `.tasm` programmatically when generated kernels become useful. Source listing is the canonical artefact; binary is derived.

---

## 9. Bridge to `ntransformer`

```
vendor/ntransformer/  (lift, surgical fork)
├── core/tensor.hpp           ← extend dtype enum: + TritFP_B; device enum: + Sim
├── model/config.{h,cpp}      ← drop-in
├── model/loader.{h,cpp}      ← patch dtype mapping: bf16/Q*/F16 → TritFP_B via quantize_to_trit
├── model/transformer.cpp     ← orchestration unchanged; replace cudaLaunch sites with sim.call_kernel
├── model/attention.cpp       ← structure unchanged; per-step calls to tk_attn_*
├── model/ffn.cpp             ← idem, calls to tk_swiglu and tk_matmul_b_9t
├── model/norm.cpp            ← idem, calls to tk_rmsnorm
├── inference/tokenizer.cpp   ← drop-in
├── inference/sampler.cpp     ← drop-in
└── cuda/*                    ← INTENTIONALLY DROPPED — replaced by src/kernels/ + src/sim/
```

The bridge is the `TernarySim` backend at `src/host/sim_backend.cpp`:

```cpp
namespace nt {
class TernarySim : public Backend {
  ter::Sim sim;
  ter::KernelTable kernels;

  void matmul(Tensor x, Tensor w, Tensor y) override {
    sim.dma_in(x.data, addr_x, x.bytes);
    sim.call_kernel(kernels.matmul, {addr_x, w.sim_addr, addr_y, M, N, K, x.scale * w.scale});
    sim.dma_out(addr_y, y.data, y.bytes);
  }
  // rmsnorm, rope, swiglu, attention_score, softmax, ...
};
}
```

`forward.cpp` from `ntransformer` does not change. The only swap is the backend device implementation, exactly as a CUDA-vs-CPU dispatch already works in `ntransformer`.

---

## 10. Validation, Metrics, Falsifiable Hypotheses

### 10.1 Hypotheses

- **H1.** For Llama 3.2 1B, the forward pass executed on the ternary sim with format B (9 trits/element, per-tensor scale) reproduces logits with mean squared error below ε (target ε ≤ 1e-3 over softmax outputs) vs a bf16 reference, while reducing the count of arithmetic operations relative to a binary fp16 equivalent.
- **H2.** Format A (`tfloat` ternary native) preserves H1 quality with fewer total trits per tensor (target ~6–7 trits/element), confirming the radix-3 information-density advantage.
- **H3.** (bonus) Under BitNet b1.58 (weights natively `{-1, 0, +1}`), the operation breakdown collapses to `tvadd` only — zero `tvmac` — demonstrating the compounded saving when substrate and data alignment match.

### 10.2 Reportable metrics (per run)

| Metric | Source | Significance |
|---|---|---|
| Operation breakdown | `OpCounters` per category | thesis headline — matmul reduces to additions |
| Trits moved through memory | bytes read/written × 5 trits/byte (ideal) | bandwidth thesis |
| Logit MSE vs reference | bf16 reference (PyTorch + meta-llama/Llama-3.2-1B) on CPU | quality gate (target MSE ≤ 1e-3 on softmax) |
| Perplexity | WikiText-2, 1k tokens | publishable standard metric (target ≤ 1.05× bf16) |
| Top-1 token agreement | vs bf16 reference, 1k tokens | sanity check |
| BLEU on TinyStories generation | self-generated continuations | qualitative humanity check |
| Comparison B vs A vs (eventual) ternary-native | full suite, three modes | central deliverable |

### 10.3 Per-phase test gates

```
F0 types     → property-based: associativity, commutativity, double-negation
F1 ISA       → golden tests: factorial(5) in .tasm produces 120 in trit form
F2 SIMD      → matmul random 64×64 vs numpy, error within quantisation noise
F3 quantize  → bf16 → trit → bf16 round-trip, MSE under predicted bound
F4 kernels   → each kernel vs ntransformer CPU/CUDA equivalent on identical input
F5 bridge    → TinyLlama (~10M params) end-to-end, top-1 agreement vs HF ≥ 95 %
F6 Llama 1B  → WikiText-2 perplexity ≤ 1.05× bf16 reference
```

Each gate is quantitative. A failing gate blocks promotion to the next phase.

---

## 11. Phasing

1. **F0 — Ternary primitives.** `Trit, Tryte, Word27, Word54`, packing, exhaustive arithmetic tests. No sim yet.
2. **F1 — ISA + assembler + scalar sim.** Implements Format B's scalar subset. Smoke: assembled `factorial.tasm` runs.
3. **F2 — SIMD extension.** `tvadd, tvmac, tvload, tvsum, tvbroadcast`. Aisolated matmul vs numpy reference.
4. **F3 — Format B converters.** Quantizer, dequantizer, round-trip tests.
5. **F4 — Bytecode kernels (K3).** `tk_matmul_b_9t, tk_rmsnorm, tk_rope, tk_swiglu, tk_softmax, tk_attn_score`. Each benchmarked vs reference.
6. **F5 — Bridge to ntransformer.** Lift and patch `vendor/ntransformer/`. `TernarySim` backend. Smoke with TinyLlama.
7. **F6 — Llama 3.2 1B end-to-end.** Full validation suite.
8. **F7 (optional)** — Format A (`tfloat`) and B-vs-A comparison.
9. **F8 (optional)** — Pure-scalar ISA path for honest TADD counting.
10. **F9 (optional)** — BitNet b1.58 case (zero-`tvmac` regime).
11. **F10 — K4: bare-metal optimisation path.** Aggressively optimise the simulator so it can be embedded as a ring-0 backend in `~/osito-k`, exploiting the binary substrate as efficiently as possible. Detailed in §11.bis.

---

## 11.bis — K4: Bare-Metal Deployment Path

K4 is the eventual destination for the simulator: not a destination model, but a deployable backend. The goal is a build of `ter` that loads inside `~/osito-k`'s ring-0 inference path and serves the existing `sys_inference` (530–534) syscall family with a ternary backend, achieving the most efficient ternary execution possible given that the substrate is binary x86-64.

### 11.bis.1 Target shape

A static library (`libter_k4.a`) linked into the osito-k kernel image, exposing:

```
ter_k4_init(arena_base, arena_size)            // bind a slab of tensor_arena
ter_k4_load_weights(weights_blob, layout)      // populate weight region
ter_k4_forward(input_ids, n, kv_cache, out)    // one forward step
ter_k4_counters(&out)                          // for /sys/kernel/ter/counters
```

It does **not** allocate via `kmalloc`/`vmalloc`; it carves its memory out of the existing `tensor_arena` (the 512 MB superpage region documented in `osito-k/docs/x86-features-detail.md`). It does not depend on libc, the heap, the scheduler, or float context save/restore (FPU/SSE/AVX) outside the kernel's existing dispatch hooks.

### 11.bis.2 Optimisation budget against binary substrate

The host simulator (F0–F6) is correctness-first. K4 is performance-first within the constraint that **semantics must remain bit-identical to the reference simulator**. Optimisation techniques in scope:

- **Trit packing for SIMD.** Pack 80 trits per ZMM register (AVX-512, 2 bpt × 32 bytes = 80 active trits + 16 unused). Implement `tvadd` and the per-lane half of `tvmac` via shuffles + masked adds + carry propagation. Target: a single ZMM `tvadd` of a 27-lane vector in fewer than 10 host instructions.
- **Carry-propagating ternary add via bitmask tricks.** Two bits per trit lets us classify trit pairs `(00, 01, 10)` against each other with `popcnt`/`pdep`/`pext` (BMI2) and produce `(sum, carry)` without per-trit branches. The exact bit-trick table is documented as part of F10.
- **Dense base-3 packing for cold weights.** Replace the 2-bpt scheme with 5-trits-per-byte (base-3 digits) for read-mostly weight tensors, regaining the 33 % memory waste. Hot paths still use 2-bpt for SIMD friendliness; conversions happen on tensor load.
- **CPUID dispatch.** Reuse osito-k's existing dispatch infrastructure (boot-time pick of AVX2 / AVX-512 / scalar). Provide three `tvmac` implementations selected at init.
- **Cache-aware kernel layouts.** Tile matmul to fit weight blocks in L2 (1 MB on Ryzen 7 5800X). KV cache laid out for streaming reads. Use osito-k's superpage tensor arena to reduce TLB pressure.
- **No floating-point in hot path.** Format B's int-trit matmul is a natural fit; the per-tile float scale is the only place we touch the FPU, and it can be deferred to batch-end. Aligns with osito-k's "no floating-point reliance" philosophy.

### 11.bis.3 Constraints inherited from osito-k

- Ring 0 — no syscalls allowed inside the library; all IO via the host kernel.
- No CRT — no `malloc`, no `printf`, no exceptions, no STL containers in hot paths. (STL is acceptable in F0–F6 host builds; K4 has its own narrow C-style API surface.)
- Stable layout under `tensor_arena` — placement of weight blocks must survive `kexec` if osito-k self-recompiles.
- Counters exported via `/proc`-style debug interface for live measurement.

### 11.bis.4 Validation for K4

- Bit-identical output vs the F0–F6 reference simulator on a fixed seed, confirmed by golden-file comparison of logits over 100 tokens.
- Operation counts unchanged from reference (the optimisation is in host instructions per ternary op, not in the ternary semantics themselves).
- Boot-time smoke test: osito-k boots with `libter_k4.a` linked, runs one forward step, prints counters to serial.
- Performance regression test: measured tokens/sec on Ryzen 7 5800X tracked over time. Baseline established at first successful K4 build.

### 11.bis.5 Why K4 lives here and not in osito-k

The ternary semantics, ISA, kernel bytecode, and number formats are properties of `ter`. osito-k provides the host environment (memory arena, dispatch, IO). K4 is the integration shim. Keeping the shim in `ter` means a single source of truth for ternary correctness; osito-k consumes a versioned static library. Cross-repo coordination is via release tags.

---

## 12. Error Handling

Bare-metal philosophy: panic loud, recover never. This is a research simulator, not a production system.

### 12.1 Sim faults

`IllegalOpcode`, `MisalignedFetch`, `TritEncodingViolation` (the reserved bit pattern `11`), `MemOutOfRange`, `Overflow27`, `DivByZero` trap to a fault handler that prints the register file, the PC, and a backwards-disassembly of the last 16 instructions, then `abort()`s the host process.

### 12.2 Quantisation overflow

A weight whose absolute value exceeds `scale × max_trit_int` is clipped, a warning is emitted, and a per-tensor counter is incremented. Persistent overflow on a particular tensor is a signal to either bump `n_trits_per_elem` or switch to channel-wise scale.

### 12.3 Kernel ABI violations

Asserted in debug builds; undefined behaviour in release.

### 12.4 Host runtime errors

No `try`/`catch`. Errors are reported to `stderr` with a host stack trace plus a sim register dump.

---

## 13. Out of Scope (YAGNI)

The following are explicitly excluded from the MVP and will be considered case-by-case post-paper:

- Multi-core / parallel sim. One ternary core. Throughput, if needed, is provided by host-level parallelism over independent sim instances.
- Bootloader / kexec / sim-resident orchestrator. The host orchestrates. Migration to K2 is the upgrade path.
- MMU, paging, memory protection. Flat address space, R/W everywhere.
- Interrupts, asynchronous IO. Synchronous, kernel-by-kernel execution.
- Training, backprop in sim. Inference only.
- JIT compilation of bytecode. Always interpreted.
- GUI / REPL. Host CLI only.
- Llama 3 8B / 70B. Memory-bound on host.
- Real ternary silicon. K4's bare-metal target is still a binary host emulating ternary.

---

## 14. Identified Risks

| Risk | Probability | Mitigation |
|---|---|---|
| Sim too slow to debug F6 | High | Start with TinyLlama in F5; instrument counters from F1 onward |
| Format B at 9 trits degrades quality below gate | Medium | Bump to 12 trits is a single constant change; channel-wise scale fallback |
| Bridge to ntransformer needs major refactor | Medium | F5 is the highest-risk phase; reserve a buffer; clean adapter keeps `forward.cpp` intact |
| LUTs erode ternary purity | Low–medium | "Honest mode" flag computes transcendentals via Newton-Raphson, reports both counters |
| Hand-written ternary asm becomes unmaintainable | Medium | C++ macros emit verifiable `.tasm` text instead of binary; source listing is canonical |

---

## 15. Open Questions Resolved During Brainstorm (for record)

- **Why balanced ternary?** Negation is per-trit inversion; symmetric rounding; historical precedent (Setun, 1958).
- **Why 27-trit word?** All-powers-of-3 hierarchy (27 = 3³) provides clean address space, comfortable instruction encoding, and a 27-lane vector register.
- **Why ditch a pure ternary kernel program (K2) for K3?** K3 lets us write the orchestrator in host C++ while keeping all hot math in ternary bytecode, which is enough to demonstrate the thesis without writing the entire transformer in ternary assembly. K2 is the future direction once kernel patterns stabilise.
- **Why not BitNet directly?** BitNet is a model-side answer to ternary; this project asks the substrate-side question. BitNet enters as the bonus case in §10.1 H3 where both substrate and weights align.

---

## 16. Definition of Done (MVP)

- F0 through F6 complete with all gates passed.
- `meta-llama/Llama-3.2-1B` produces coherent text generation under the simulator.
- Operation-count report and quality metrics match or exceed the targets in §10.
- `docs/` contains complete ISA, memory map, number formats, and kernel bytecode references.
- Reproducible build: `cmake -S . -B build && cmake --build build && ./build/ter run --model llama-3.2-1b --prompt "..."` succeeds on a fresh checkout.
