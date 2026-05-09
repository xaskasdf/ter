# `ter` — Kernel Authoring Patterns

Captured after F4-partial (one kernel: `tk_matmul_b_9t`). These are the conventions every K3 kernel should follow.

## Calling Convention

| Register | Role | Lifetime across `call_kernel` |
|---|---|---|
| `R0` | Hard-zero | always 0 |
| `R1..R7` | Argument and return | overwritten by caller per call |
| `R8..R15` | Caller-saved scratch | **reset to 0 at call start** |
| `R16..R25` | Callee-saved | preserved across calls |
| `R26` | Stack pointer (post-increment on push) | preserved across calls |
| `V0..V8` | Vector scratch | **reset to 0 at call start** |
| `A0..A2` | Accumulator scratch | **reset to 0 at call start** |
| `PC, halted` | Control state | set by `call_kernel`; kernel halts via `thalt` |

`Sim::call_kernel` invokes `RegFile::reset_caller_saved()` before pushing args. This means kernels can assume `A0 = 0`, `V0..V8 = 0`, `R8..R15 = 0` at entry — no need to clear them by hand. **Conversely, do not rely on V/A state from a previous call.**

## The Float-Scale Boundary

Inside the simulator, **everything is integer ternary**. The only place float arithmetic happens is on the host, applied as a post-processing scalar to the kernel's int-acc output.

For format-B matmul:
- Kernel returns: `int_acc = sum_k X.payload[i,k] * W.payload[k,j]` (pure trit math).
- Host computes: `Y[i,j] = int_acc * X.scale * W.scale`.

This is the operational embodiment of the project's thesis. **Never load or compute floats inside a kernel.** Transcendentals (sigmoid, exp, sin, cos, rsqrt) live in lookup tables that store fixed-point integer trits — see "Transcendentals via LUT" below.

## Single-Tile vs Internal-Loop Kernels

`tk_matmul_b_9t` is a **single-tile** kernel: it computes one 27-length dot product. Multi-tile work (K > 27, or M > 1, or N > 1) is orchestrated by the host, which invokes the kernel once per tile.

Trade-offs:

- **Single-tile (current):** simpler kernel; host pays one `call_kernel` overhead per tile; A0 reset between calls means each tile is independent.
- **Internal-loop (future):** kernel takes K, M, N as args and loops internally; one `call_kernel` per matmul; needs `tjump`-based loop control and address arithmetic in trits.

Default to single-tile for new kernels until the host orchestration becomes the bottleneck. When migrating to internal-loop, document the per-call invariants explicitly.

## Scratch Buffer Pattern

When the kernel needs a contiguous block of memory but the source data is strided (e.g., gathering a column from a row-major matrix), copy through a fixed scratch region first.

In `test_kernel_matmul_b_32.cpp`:
- Source A is row-major `(M, K)` — row `i` is contiguous.
- Source B is row-major `(K, N)` — column `j` is **strided** by `N`.
- Per output cell, the host copies A's row chunk and B's column chunk into scratch buffers at fixed addresses (`scratch_x = 9000`, `scratch_w = 9100`), then passes those addresses to the kernel.

Convention: reserve a "scratch" region in the sim memory map and document its base address. Don't reuse data-path memory for scratch.

## Transcendentals via LUT

For kernels that need non-linear functions (sigmoid in SwiGLU, exp in softmax, rsqrt in RMSNorm, cos/sin in RoPE), use **lookup tables stored in sim memory** as integer trits. Pattern:

1. Host generates the LUT once (during sim init): `lut[i] = round(f(i * step) / scale_y) * mti_per_elem`.
2. Host loads the LUT into a fixed memory region.
3. Kernel reads the LUT entry by computing the index and `tload`ing.
4. Final scaling lives on the host (same float-scale boundary as matmul).

LUT size budget: 256 entries (~256 Word27s = ~5 KB packed) for cheap functions; 2048 for high-precision. The "honest mode" debug flag (in the spec) will let us replace LUTs with Newton-Raphson kernels for honest counter measurements; for now LUT is the default.

## Operation Counters Are the Headline

Every kernel test should `CHECK` the OpCounters at the end. Counts are the publishable artefact and a great regression catch (e.g., counter going up when it shouldn't = unintended kernel invocation).

For matmul: `M * N * ceil(K / 27)` `tvmac`s, `M * N * ceil(K / 27)` `tvsum`s, `M * N * ceil(K / 27)` `tstore`s, plus per-call overhead `tvload`s.

## Build & Test Wiring

- Each kernel `.tasm` lives at `src/kernels/tk_<name>.tasm`.
- `src/kernels/registry.cpp` reads + assembles all defaults at sim init.
- Tests link against the absolute kernels dir via `target_compile_definitions(... PRIVATE TER_KERNELS_DIR="...")`.
- Tests with float-reference data declare a CMake fixture using `.venv/bin/python` and a tools script.

Keep this file up to date as new patterns emerge.

## Lessons from `tk_rope` (host-prepared inputs + wide-clamp consistency)

### Host-prepared inputs unlock pure SIMD
RoPE's pair-rotation `(x[2k], x[2k+1]) → (x[2k]*c - x[2k+1]*s, x[2k]*s + x[2k+1]*c)` would need lane-shuffle and pair-negate opcodes to compute in-kernel. Building `cos_vec`, `sin_vec`, and `rotated_x` on the host ahead of `call_kernel` reduces the kernel to 8 instructions: `4 tvload + 2 tvmul + 1 tvadd + 1 tvstore`. Counter signature is exact and the kernel is trivially auditable.

This joins **host-orchestrated tiling** (matmul) and **host-side scale recovery** (every kernel) as the third architectural separation between kernel and host.

### Wide-clamp consistency across all SIMD ops
Originally `Vec::set_lane` clamped to `±9841` (the 9-trit lane limit) and `set_lane_wide` clamped to int32 (added in F4.6.3 for `TVMUL` so products like `9841² ≈ 9.7e7` wouldn't saturate). This left `TVADD`, `TVSUB`, `TVNEG` still using the narrow clamp — fine in isolation but **catastrophic when chained after `TVMUL`** (RoPE: `tvadd v6, v4, v5` where v4 and v5 are `tvmul` outputs).

The unified semantic (post-fix): **all SIMD ops use `set_lane_wide`**, which clamps at int32 boundary. The "9-trit per lane" guarantee applies to memory boundaries (loads from `Word27` in memory) and the per-tensor scale recovery on the host, not to in-register intermediates.

**Implication for new kernels:** intermediates can grow up to int32 (~2.1e9). Sums of two `tvmul` products (≈ `2 × 9841² ≈ 1.94e8`) fit comfortably. Triples (`9841³ ≈ 9.5e11`) still overflow — the SwiGLU/SiLU split-at-host pattern is still required.

### When the recovery formula is "trivial"
For pure-linear kernels (only multiplies and adds on quantized inputs and pre-scaled coefficients), the recovery formula is just the product of the input scales divided by the coefficient's `OUT_SCALE`. RoPE: `recovery = xt.scale / OUT_SCALE`. No exp/rcp/rsqrt cancellations. This is the simplest possible recovery and a good signal that a kernel is well-shaped.

## Lessons from `tk_silu` (memory-aliasing bug)

### Reserve the low memory range for kernel code
Without a memory-map convention, tests put data at convenient round addresses (100, 200, 300...) which silently overlap with kernel code installed by `KernelTable::install`. With 4 installed kernels totaling ~120 instructions, code occupies addresses 0–120; data at addr 100 overwrites the tail of the last installed kernel; subsequent kernel calls execute garbage and either loop forever or trigger memory faults from corrupted load addresses.

**Convention (effective immediately):** reserve addresses `[0, 511]` for kernel code (gives ~5 kernels worth of headroom). Data, scratch, and LUTs go at `>= 512`. The convention numbers in current tests:
- code: `0..511`
- data inputs: `>= 512` (most tests use 1000+)
- LUTs: `>= 4096` (one LUT-aligned region per LUT)
- scratch (kernel-internal): `700..899` (this region is below the new convention floor — needs to move to `>= 4096` in any new kernel; existing kernels keep `700` for now and will be migrated when the next bug appears)

The robust fix is a host-side memory allocator that hands out non-overlapping regions; for now the convention is enforced by code review and tests use addresses `>= 1000`.

### When tests fail with `Memory::load_word out_of_range`, suspect aliasing first
The failure mode is sneaky: the kernel runs, decoded `tvload v0, r1` reads bytes from a corrupted instruction word as a "vector address," and the result is a huge garbage address that throws `out_of_range`. Stack trace points at the load, not the corruption site. Always verify data addresses are above the kernel-code high-water mark before debugging the kernel itself.

### Counter pattern for composed-kernel tests
For SwiGLU we wrote two TEST_CASEs:
1. `tk_silu` direct vs numpy `silu(gate)` — verifies the kernel.
2. Full SwiGLU = `silu(gate) * up` via kernel + host `tvmul` — verifies the composition.

Both passing means the composition pattern is correct AND the kernel is correct. Either failing isolates the bug.

## Lessons from `tk_softmax` (per-lane LUT lookup + recovery math)

### Per-lane lookup is unavoidable without lane extract/insert opcodes
The kernel walks 27 iterations of a software loop per call: read lane scalar from memory, compute index via signed division, `tload` from the per-element LUT, write to scratch buffer. Total per-call: ~150-200 instructions executed. Slow per-tile but correct, and counters report all of it.

A future `TVEXTRACT/TVINSERT` opcode pair would let us SIMD-ify this: broadcast inputs to a vreg, do per-lane LUT via gather (would need a `TVGATHER` op too), write results back. Documented as future improvement.

### Signed division by repeated subtraction
Quantized inputs are signed (`x_int ∈ [-9841, 9841]`). The kernel branches on `tblt r18, r0, neg_div` to dispatch positive vs negative paths. Both use repeated subtraction; the negative path subtracts `r5` until the running value crosses zero from below.

### Recovery formula derivation (the bug we hit)
**This was wrong in the original plan.** Correct math for softmax:

- Inputs: `x_int[i]` quantized with scale `xt.scale` such that `x[i] ≈ x_int[i] * xt.scale`.
- exp LUT entry: `exp_lut[ei] ≈ exp(x[i]) * OUT_SCALE / exp_max`.
- sum_E (kernel int): `sum_E ≈ sum_i(exp_lut[ei]) ≈ sum_exp * OUT_SCALE / exp_max`.
- rcp index from `sum_div = N * OUT_SCALE / 255`: `rcp_idx ≈ sum_E / sum_div = sum_exp * 255 / (exp_max * N)`.
- rcp LUT entry: `rcp_lut[ri] ≈ (1 / value_at_rcp_idx) * OUT_SCALE / rcp_max`. Since `value_at_rcp_idx = (rcp_idx + 1) / 256 ≈ rcp_idx / 256`, this simplifies to: `rcp_lut[ri] ≈ 256 * OUT_SCALE / (rcp_max * rcp_idx) ≈ exp_max * N * OUT_SCALE / (sum_exp * 255 * rcp_max / 256)`.
- Product: `y_int[i] = exp_lut[ei] * rcp_lut[ri] ≈ exp(x[i]) / sum_exp * OUT_SCALE^2 * N / 255` (the `exp_max` and `rcp_max` factors cancel out).
- **Recovery: `y[i] ≈ y_int[i] * 255 / (OUT_SCALE^2 * N)`.**

The cancellation of `exp_max` and `rcp_max` is non-obvious and was missed in the spec. **Always derive the recovery formula from first principles per kernel and verify on one element by hand before committing the test.**

### Counter signature
For `tk_softmax`: 1 TVMUL, 2 TVLOAD, 1 TVSTORE, plus the per-lane scalar work: ~55 TLOAD (27 x reads + 27 exp lookups + 1 rcp lookup), 27 TSTORE for scratch_E, 27 TSTORE for scratch_bcast, plus the loop bookkeeping (TADDs, TBLTs, TJUMPs, TLOADIs). Counter checks should pin the SIMD ones exactly and use `>=` for the per-lane scalars.

## Lessons from `tk_rmsnorm` (and three bugs found)

These bugs were found while writing the first jumping kernel. Documenting so they don't regress.

### Jump relocation in `KernelTable::install`

The assembler resolves labels to **blob-relative indices** (label `loop` at the start of the blob = address 0). When the kernel is installed at `entry_addr > 0`, those immediates need to be shifted by `entry_addr` before being written to memory. `KernelTable::install` does this by patching TBEQ/TBNE/TBLT/TJUMP/TCALL immediate fields at install time.

The first kernel ever installed (`tk_matmul_b_9t`) had no jumps, so this latent bug was invisible until `tk_rmsnorm` introduced the first looping kernel. The regression test in `tests/test_kernel_relocation.cpp` covers the install-time patcher explicitly.

### `TVMUL` saturation: lane width vs intermediate

Originally `Vec::set_lane` clamped to `±9841` (the 9-trit per-lane range). For `TVMUL`, intermediate products like `9841 × 9841 ≈ 1e8` saturate immediately. The fix introduces `Vec::set_lane_wide` which clamps only at the `int32_t` boundary, used by `vec_mul`. `TVMAC` still uses the int64 accumulator (acc lanes are int64 already).

**Implication:** TVMUL outputs are wider than 9 trits per lane. Code that stores them back to memory via `tvstore` will lose the high bits if the destination expects 9-trit semantics. Always quantize/scale outputs before storage when chaining kernels.

### Recovery formula clarity

For RMSNorm, the recovery scale that converts the kernel's int output back to float is `rsq_max / (OUT_SCALE * mti)` (NOT `xt.scale * rsq_max / OUT_SCALE` as initially documented). Each LUT-based kernel needs its recovery formula derived explicitly and tested. Document it in the kernel's source comment.

## F4 Complete — Architectural Summary

After 5 implemented kernels and one host-composed attention pipeline, three architectural seams have proven essential:

1. **Host orchestrates tiling.** Long matmuls split into 27-length chunks; the host loops, the kernel does one tile.
2. **Host applies scale recovery.** All in-kernel arithmetic is integer ternary; per-tensor scales live on the host and apply at boundaries.
3. **Host prepares non-uniform inputs.** Pair-shuffles, broadcast vectors, and any structure the SIMD ISA can't easily produce live in host code that runs once before the kernel call.

Composition (e.g., attention) is just kernel calls in sequence. No "mega-kernel" was needed for any transformer op. Total kernel count: 5 mechanical kernels + 1 attention orchestration pattern.

### Counter taxonomy
- `tk_matmul_b_9t`: 1 TVMAC, 1 TVSUM, 1 TSTORE per tile
- `tk_rmsnorm`: 1 TVMAC + 1 TVSUM + 1 TVMUL + ~30 TLOAD per call (per-lane broadcast loop)
- `tk_softmax`: 1 TVMUL + 2 TVLOAD + 1 TVSTORE + ~55 TLOAD per call (per-lane exp loop)
- `tk_silu`: 1 TVMUL + 2 TVLOAD + 1 TVSTORE + ~54 TLOAD per call (per-lane sigmoid loop)
- `tk_rope`: 4 TVLOAD + 2 TVMUL + 1 TVADD + 1 TVSTORE per call (pure SIMD)

The total TVMAC count for a single attention pass is the headline metric for the project's thesis: a count we can compare against an equivalent fp16 multiply count.

### What we'd build next if continuing F4
- `tk_attn_score` as a single sim-resident composition kernel (would use `tcall` to chain other kernels' entry points). Currently this lives in host code; making it sim-resident would close the K3 → K2 migration.
- `TVEXTRACT/TVINSERT/TVGATHER` opcodes to SIMD-ify the per-lane LUT loops in softmax/silu.
- `TVCLEAR` opcode to explicitly zero an accumulator (currently relied on `call_kernel` reset).

These are quality-of-life improvements; the core thesis is provable as-is.
