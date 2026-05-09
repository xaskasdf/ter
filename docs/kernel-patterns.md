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
