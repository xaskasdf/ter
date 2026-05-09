# `ter` Softmax Kernel Plan (F4-cont)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Implement `tk_softmax` — the second transcendental kernel — using a host-generated `exp` LUT (per-lane lookup via memory loop) and a `recip` LUT for the normalisation. End state: random input vector → softmax via ternary kernel → matches numpy reference within bounded relative error. Establishes the per-lane LUT lookup pattern.

**Architecture:** Kernel reads input vector from memory; for each of 27 lanes, computes a LUT index from the lane value and `tload`s `exp_lut[idx]`, accumulates the running sum and writes to a scratch buffer. Then divides the sum into a reciprocal LUT index, loads `rcp_lut[idx2]`, broadcasts it via the same store-then-tvload pattern used in `tk_rmsnorm`, and `tvmul`s element-wise.

**Tech Stack:** Same as plan 3.

**Spec:** `docs/superpowers/specs/2026-05-08-ter-design.md` §8 (kernel catalogue). Patterns: `docs/kernel-patterns.md`.

**Out of scope:** Numerical stability via max-subtract (deferred — quantized 9-trit inputs are bounded). SwiGLU, RoPE, attention. F5 bridge.

---

## Why this design

Softmax `y_i = exp(x_i) / sum_j(exp(x_j))` has two transcendentals: `exp` per lane, and `1/sum` once. Two LUTs, in two phases:

1. **Per-lane exp loop** — kernel walks 27 lanes one at a time (no SIMD), reading `mem[x_addr + k]`, computing the LUT index from the lane value, looking up `exp_lut[idx]`, accumulating into a running sum, and writing the exp value to a scratch slot. Cost per lane: ~6 trit instructions. Total: ~162 + bookkeeping.
2. **Reciprocal + scale** — divide sum into recip LUT index (repeated subtraction), look up `rcp_int`, broadcast it to a vreg via the store-then-tvload pattern (same wart as rmsnorm), `tvload` the exp scratch into a vreg, `tvmul`, `tvstore` outputs.

This is **not the SIMD-optimal** softmax (we'd need lane extract/insert opcodes for that). It IS the cleanest given current ISA. A future `TVEXTRACT/TVINSERT` opcode pair would let us SIMD-ify the exp lookup; documented as a follow-up improvement.

### Numerical stability

Skipped for this MVP: no max-subtract. Input is 9-trit quantized so values are bounded `[-9841, +9841]`. The exp LUT is sized to cover this range; large positive inputs will saturate to the LUT's max entry. For a small softmax over the 27-element tile, this is acceptable. For real attention with large logits, max-subtract becomes mandatory — handled in the attention kernel plan.

---

## File Structure (created during this plan)

```
tools/
└── gen_softmax_luts.py        # exp_lut.bin + rcp_lut.bin + meta files
src/kernels/
└── tk_softmax.tasm
tests/
└── test_kernel_softmax.cpp
docs/
└── (kernel-patterns updated; isa unchanged)
```

---

## Task F4.7.1 — `gen_softmax_luts.py` (exp + recip LUTs)

**Files:** Create `tools/gen_softmax_luts.py`.

The exp LUT maps `idx ∈ [0, 255]` to `round(exp(x_at_idx) * out_scale / exp_max)` where `x_at_idx = (idx - 127.5) / 32` (so input range ≈ ±4 in continuous units, mapped to ~±9000 in 9-trit input via `idx = floor((trit_value + max_trit/2) / step)`). The recip LUT maps `idx ∈ [0, 255]` to `round((1 / value_at_idx) * out_scale)` where `value_at_idx = (idx + 1) / N_ENTRIES` to avoid division by zero.

For this MVP, fix `out_scale = 9841` (max 9-trit int) and `N_ENTRIES = 256`.

- [ ] **Step 1: Write `tools/gen_softmax_luts.py`**

```python
#!/usr/bin/env python3
"""Generates exp + reciprocal LUTs for tk_softmax."""
import argparse
import os
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="build/lut_data")
    p.add_argument("--n-entries", type=int, default=256)
    p.add_argument("--out-scale", type=int, default=9841)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # exp LUT: covers x ∈ [-4, +4] in continuous units, mapped to idx [0, 255].
    idx = np.arange(a.n_entries, dtype=np.float64)
    x = (idx - (a.n_entries - 1) / 2.0) / 32.0          # x ∈ [-4, +4]
    e = np.exp(x)
    exp_max = e.max()
    exp_lut = np.round(e * a.out_scale / exp_max).astype(np.int32)
    exp_path = os.path.join(a.out_dir, "exp_lut.bin")
    exp_lut.tofile(exp_path)

    # recip LUT: covers value ∈ [1/N, 1] (avoid 0), maps idx → round(1/value * out_scale).
    val = (idx + 1.0) / float(a.n_entries)
    r = 1.0 / val
    rcp_max = r.max()
    rcp_lut = np.round(r * a.out_scale / rcp_max).astype(np.int32)
    rcp_path = os.path.join(a.out_dir, "rcp_lut.bin")
    rcp_lut.tofile(rcp_path)

    with open(os.path.join(a.out_dir, "softmax_lut.meta"), "w") as f:
        f.write(f"n_entries={a.n_entries}\n")
        f.write(f"out_scale={a.out_scale}\n")
        f.write(f"exp_max={exp_max}\n")
        f.write(f"rcp_max={rcp_max}\n")
        f.write(f"x_step=0.03125\n")              # 1/32
        f.write(f"x_offset_idx={(a.n_entries - 1) / 2.0}\n")
    print(f"wrote exp_lut.bin and rcp_lut.bin to {a.out_dir}")
    print(f"exp_max = {exp_max:.4f}, rcp_max = {rcp_max:.4f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test**

```
chmod +x tools/gen_softmax_luts.py
.venv/bin/python tools/gen_softmax_luts.py --out-dir build/lut_data
ls -la build/lut_data/
```

Expected: `exp_lut.bin` (1024 B), `rcp_lut.bin` (1024 B), `softmax_lut.meta` exist.

- [ ] **Step 3: Commit**

```
git add tools/gen_softmax_luts.py
git commit -m "feat(tools): exp + recip LUT generator for tk_softmax"
```

---

## Task F4.7.2 — `tk_softmax.tasm` (the kernel)

**Files:** Create `src/kernels/tk_softmax.tasm`. Modify `src/kernels/registry.cpp`.

Kernel signature:
```
tk_softmax(addr_X, addr_Y, addr_EXP_LUT, addr_RCP_LUT, x_scale_div, sum_div):
  - r1 = addr_X (27 input Word27s)
  - r2 = addr_Y (27 output Word27 slots)
  - r3 = addr_EXP_LUT (256 entries)
  - r4 = addr_RCP_LUT (256 entries)
  - r5 = x_scale_div (per-call divisor: x_int / x_scale_div + 128 → exp idx)
  - r6 = sum_div (per-call divisor: sum_int / sum_div → rcp idx)
```

Algorithm:
```
sum = 0
for k in 0..27:
  x_int = mem[addr_X + k]
  idx = clamp(x_int / x_scale_div + 128, 0, 255)
  e_int = mem[addr_EXP_LUT + idx]
  mem[scratch_E + k] = e_int
  sum += e_int

rcp_idx = clamp(sum / sum_div, 0, 255)
rcp_int = mem[addr_RCP_LUT + rcp_idx]

# Broadcast rcp_int into V1 (store-then-tvload pattern)
for k in 0..27:
  mem[scratch_bcast + k] = rcp_int

V0 = tvload(scratch_E)
V1 = tvload(scratch_bcast)
V2 = V0 * V1
tvstore V2 to addr_Y
halt
```

Use `r10 = const 1`, `r11 = loop counter`, `r12 = const 27`, `r13 = const 128`. Scratch addresses: `r14 = 700` (for E), `r15 = 800` (for bcast). Constant `r16 = sum running` (callee-saved so it persists across the loop iterations within the kernel).

- [ ] **Step 1: Write `src/kernels/tk_softmax.tasm`**

```
        ; tk_softmax — 27-element softmax via per-lane exp lookup + scale.
        ; Args:
        ;   r1 = addr_X
        ;   r2 = addr_Y
        ;   r3 = addr_EXP_LUT (256 entries)
        ;   r4 = addr_RCP_LUT (256 entries)
        ;   r5 = x_scale_div (x_int / r5 + 128 = exp_idx)
        ;   r6 = sum_div (sum_int / r6 = rcp_idx)

        ; Constants
        tloadi      r10, 1
        tloadi      r12, 27
        tloadi      r13, 128
        tloadi      r14, 700      ; scratch_E base
        tloadi      r15, 800      ; scratch_bcast base
        tloadi      r16, 0        ; running sum
        tloadi      r11, 0        ; loop counter

        ; Per-lane exp loop:
        ;   x_int = mem[r1 + r11]
        ;   idx_signed = x_int / r5
        ;   exp_idx = clamp(idx_signed + 128, 0, 255)
        ;   e_int = mem[r3 + exp_idx]
        ;   mem[r14 + r11] = e_int
        ;   r16 += e_int
        ;   r11 += 1; if r11 < 27 loop

exp_loop:
        ; addr_x = r1 + r11
        tadd        r17, r1, r11
        tload       r18, r17        ; r18 = x_int

        ; idx_signed = r18 / r5  (signed division by repeated subtraction; r5 > 0)
        ; For x_int < 0: subtract from 0 instead.
        tloadi      r19, 0          ; idx_signed
        tsign       r20, r18        ; r20[0] = sign of r18 (-1, 0, +1)
        tblt        r18, r0, neg_div
        ; positive division: while r18 >= r5, r18 -= r5; r19 += 1
pos_div:
        tblt        r18, r5, after_div
        tsub        r18, r18, r5
        tadd        r19, r19, r10
        tjump       pos_div
neg_div:
        ; negative: while r18 + r5 <= 0, r18 += r5; r19 -= 1
        tadd        r21, r18, r5
        tblt        r0, r21, after_div   ; if 0 < r18 + r5, stop
        tadd        r18, r18, r5
        tsub        r19, r19, r10
        tjump       neg_div
after_div:
        ; exp_idx = r19 + 128  (clamped to [0, 255])
        tadd        r19, r19, r13
        tblt        r19, r0, idx_zero
        tloadi      r22, 256
        tblt        r19, r22, idx_ok
        tsub        r19, r22, r10        ; r19 = 255
        tjump       idx_ok
idx_zero:
        tloadi      r19, 0
idx_ok:
        ; e_int = mem[r3 + r19]
        tadd        r17, r3, r19
        tload       r23, r17        ; r23 = e_int

        ; mem[r14 + r11] = r23
        tadd        r17, r14, r11
        tstore      r23, r17

        ; r16 += r23 (running sum)
        tadd        r16, r16, r23

        ; r11 += 1; loop if r11 < 27
        tadd        r11, r11, r10
        tblt        r11, r12, exp_loop

        ; --- After exp loop: r16 = sum, scratch_E populated ---

        ; rcp_idx = r16 / r6, clamp to [0, 255]
        tloadi      r19, 0
rcp_div:
        tblt        r16, r6, rcp_done
        tsub        r16, r16, r6
        tadd        r19, r19, r10
        tjump       rcp_div
rcp_done:
        tloadi      r22, 256
        tblt        r19, r22, rcp_ok
        tsub        r19, r22, r10
rcp_ok:
        ; rcp_int = mem[r4 + r19]
        tadd        r17, r4, r19
        tload       r24, r17        ; r24 = rcp_int

        ; Broadcast r24 into 27 mem slots starting at r15
        tloadi      r11, 0
bcast_loop:
        tadd        r17, r15, r11
        tstore      r24, r17
        tadd        r11, r11, r10
        tblt        r11, r12, bcast_loop

        ; V0 = exp values; V1 = rcp broadcast; V2 = V0 * V1; store to Y
        tvload      v0, r14
        tvload      v1, r15
        tvmul       v2, v0, v1
        tvstore     v2, r2

        thalt
```

This is **long** (~80 lines, ~50 instructions when assembled, ~135 instruction-executions per call due to the 27-iteration loops). Worth the size for the correctness clarity.

- [ ] **Step 2: Update `src/kernels/registry.cpp`**

After the existing `tk_rmsnorm` install, add:
```cpp
    auto src_sm = read_text(kernels_dir + "/tk_softmax.tasm");
    auto blob_sm = assemble(src_sm);
    kt.install(sim, "tk_softmax", blob_sm);
```

- [ ] **Step 3: Build and verify the kernel assembles**

```
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: 32/32 (no new test, but install must succeed). If the assembler rejects something, debug.

- [ ] **Step 4: Commit**

```
git add src/kernels/tk_softmax.tasm src/kernels/registry.cpp
git commit -m "feat(kernels): tk_softmax with per-lane exp + recip LUT lookup"
```

---

## Task F4.7.3 — End-to-end softmax test vs numpy

**Files:** Create `tests/test_kernel_softmax.cpp`. Modify `tests/CMakeLists.txt`.

- [ ] **Step 1: Write `tests/test_kernel_softmax.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/numfmt.hpp>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <string>

#ifndef TER_KERNELS_DIR
#error "TER_KERNELS_DIR must be defined"
#endif

using namespace ter;

static std::vector<int32_t> read_i32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.is_open());
    f.seekg(0, std::ios::end);
    size_t n = static_cast<size_t>(f.tellg()) / 4;
    f.seekg(0);
    std::vector<int32_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4));
    return v;
}

TEST_CASE("tk_softmax matches numpy reference within bounded rel_err") {
    constexpr int N = 27;
    constexpr int N_ENTRIES = 256;
    constexpr int OUT_SCALE = 9841;

    // Bounded random input (small range to avoid LUT saturation).
    std::vector<float> x(N);
    std::mt19937 rng(0xCAFE);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& v : x) v = dist(rng);

    // Numpy reference: y = exp(x) / sum(exp(x))
    std::vector<double> ex(N);
    double sum_e = 0.0;
    for (int i = 0; i < N; ++i) { ex[i] = std::exp(double(x[i])); sum_e += ex[i]; }
    std::vector<float> y_ref(N);
    for (int i = 0; i < N; ++i) y_ref[i] = static_cast<float>(ex[i] / sum_e);

    TritTensor xt = quantize(x.data(), {N}, 9);

    // Load both LUTs.
    auto exp_i32 = read_i32("lut_data/exp_lut.bin");
    auto rcp_i32 = read_i32("lut_data/rcp_lut.bin");
    REQUIRE(exp_i32.size() == N_ENTRIES);
    REQUIRE(rcp_i32.size() == N_ENTRIES);

    std::vector<int> exp_lut(exp_i32.begin(), exp_i32.end());
    std::vector<int> rcp_lut(rcp_i32.begin(), rcp_i32.end());

    // Read meta for recovery scales.
    float exp_max = 0.0f, rcp_max = 0.0f, x_step = 0.0f;
    {
        std::ifstream meta("lut_data/softmax_lut.meta");
        REQUIRE(meta.is_open());
        std::string line;
        while (std::getline(meta, line)) {
            if (line.rfind("exp_max=", 0) == 0) exp_max = std::stof(line.substr(8));
            else if (line.rfind("rcp_max=", 0) == 0) rcp_max = std::stof(line.substr(8));
            else if (line.rfind("x_step=", 0) == 0) x_step = std::stof(line.substr(7));
        }
    }
    REQUIRE(exp_max > 0.0f);
    REQUIRE(rcp_max > 0.0f);
    REQUIRE(x_step > 0.0f);

    // Sim setup.
    Sim s(8192);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    KernelId id = kt.find("tk_softmax");
    REQUIRE(id.valid);

    int x_addr = 100, y_addr = 200, exp_lut_addr = 1000, rcp_lut_addr = 2000;
    for (int i = 0; i < N; ++i) s.mem().store_word(static_cast<size_t>(x_addr + i), xt.payload[i]);
    s.load_lut(exp_lut_addr, exp_lut);
    s.load_lut(rcp_lut_addr, rcp_lut);

    // x_scale_div: x_int = x / xt.scale; exp_idx_signed = x_int / x_scale_div = x / step
    // We want x / step → idx_signed (same as continuous x / x_step but in trit-int domain).
    // So x_scale_div = step / xt.scale.
    int64_t x_scale_div = static_cast<int64_t>(std::round(x_step / xt.scale));
    if (x_scale_div < 1) x_scale_div = 1;

    // sum_div: max sum = N * max(exp_lut) = 27 * 9841 ≈ 266k. Map to [0, 255]:
    int64_t sum_div = (N * static_cast<int64_t>(OUT_SCALE)) / 255;

    std::vector<int64_t> args = {x_addr, y_addr, exp_lut_addr, rcp_lut_addr,
                                 x_scale_div, sum_div, 0};
    s.call_kernel(kt, id, args);

    // Read y_int.
    std::vector<int> y_int(N);
    for (int i = 0; i < N; ++i) y_int[i] = static_cast<int>(s.mem().load_word(static_cast<size_t>(y_addr + i)).to_int());

    // Recovery: y_int ≈ (exp(x_i) * out_scale / exp_max) * (1 / sum * out_scale / rcp_max)
    //                 = exp(x_i) / sum * (out_scale^2) / (exp_max * rcp_max)
    // So recovery = exp_max * rcp_max / (out_scale * out_scale)
    float recovery = exp_max * rcp_max / (static_cast<float>(OUT_SCALE) * static_cast<float>(OUT_SCALE));
    std::vector<float> y(N);
    for (int i = 0; i < N; ++i) y[i] = static_cast<float>(y_int[i]) * recovery;

    // Bounded rel_err. With 256-entry exp LUT + repeated-subtraction division +
    // 256-entry recip LUT, accept up to 10% per element.
    double max_rel = 0.0;
    for (int i = 0; i < N; ++i) {
        double ref = y_ref[i];
        double got = y[i];
        double denom = std::max(1e-3, std::fabs(ref));
        double rel = std::fabs(got - ref) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    CHECK(max_rel < 1e-1);

    // Counter check: 1 TVMUL (final scale), 1 TVLOAD x 2 (E and bcast), 1 TVSTORE.
    CHECK(s.counters().get(Opcode::TVMUL) == 1);
    CHECK(s.counters().get(Opcode::TVLOAD) == 2);
    CHECK(s.counters().get(Opcode::TVSTORE) == 1);

    // Per-lane TLOADs from the exp loop and the rcp lookup: 27 (x reads) +
    // 27 (exp lookups) + 1 (rcp lookup) = 55. Plus the broadcast loop's stores.
    // Don't pin this exactly; just check it's in a reasonable range.
    CHECK(s.counters().get(Opcode::TLOAD) >= 55);
}
```

- [ ] **Step 2: Wire CMake fixture in `tests/CMakeLists.txt` (append):**

```cmake
add_test(NAME gen_softmax_luts
    COMMAND ${CMAKE_SOURCE_DIR}/.venv/bin/python
            ${CMAKE_SOURCE_DIR}/tools/gen_softmax_luts.py
            --out-dir ${CMAKE_BINARY_DIR}/lut_data
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(gen_softmax_luts PROPERTIES FIXTURES_SETUP softmax_luts)

ter_add_test(test_kernel_softmax)
target_compile_definitions(test_kernel_softmax PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
set_tests_properties(test_kernel_softmax PROPERTIES
    FIXTURES_REQUIRED softmax_luts
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
```

- [ ] **Step 3: Build and run**

Expected: 34/34 (32 prior + new fixture + new test). All green.

If `max_rel >= 1e-1`:
- Print `y` and `y_ref` to compare.
- Common bug sources: x_scale_div formula off, sign-handling in pos/neg division, idx clamp range, recovery formula.
- The kernel's signed division (pos_div/neg_div branches) is non-trivial — verify by hand for a single positive and a single negative input.

If the test fails, debug with `r19` (exp_idx) and `r24` (rcp_int) inspection — add `tstore r19, <addr>` temporarily to expose these values to the test.

- [ ] **Step 4: Commit**

```
git add tests/test_kernel_softmax.cpp tests/CMakeLists.txt
git commit -m "test(kernels): F4.7 -- tk_softmax matches numpy within 10% rel_err"
```

After: `git log --oneline -3`.

---

## Final Task — Update README + patterns

**Files:** Modify `README.md`, `docs/kernel-patterns.md`.

- [ ] **Step 1: Update `README.md` Status block**

Replace the F4 line with:
```markdown
- [x] F4 (matmul + rmsnorm + softmax) — Sim::call_kernel, KernelTable, 3 kernels (matmul, rmsnorm, softmax), TVMUL opcode, jump relocation, exp + recip + rsqrt LUTs.
- [ ] F4 (rest) — SwiGLU, RoPE, attention.
```

- [ ] **Step 2: Append a "Lessons from softmax" section to `docs/kernel-patterns.md`**

```markdown
## Lessons from `tk_softmax` (per-lane LUT lookup)

- **Per-lane lookup is unavoidable in trit asm without lane extract/insert opcodes.** The kernel walks 27 iterations of a software loop, doing 5-7 trit instructions per lane plus 1 `tload`. Total per-call: ~150-200 instructions executed inside one `call_kernel`. Slow per-tile but correct.
- **Signed division** (`x_int / x_scale_div` with possibly negative `x_int`) requires branching on the sign first. The kernel uses `tblt r18, r0, neg_div` to dispatch positive vs negative paths. Both paths use repeated subtraction; the negative path subtracts toward zero.
- **Two LUTs in one kernel** (exp + recip) means two address-base args. Convention: pack closely — `addr_EXP_LUT, addr_RCP_LUT` consecutive in the arg list. Document each LUT's domain and recovery scale in a header comment of the .tasm file.
- **A `TVEXTRACT/TVINSERT` opcode pair** would let us SIMD-ify the exp loop (broadcast x into a vreg, do per-lane LUT via gather, write back). Listed in `docs/isa.md` as a future improvement.
- **Numerical stability** (max-subtract before exp) was deferred. For the attention kernel where logits can be large, max-subtract becomes mandatory and adds: 1 reduction (extract max from V0), 1 broadcast, 1 vsub.
```

- [ ] **Step 3: Build and verify (no test changes)**

Expected: 34/34 still.

- [ ] **Step 4: Commit**

```
git add README.md docs/kernel-patterns.md
git commit -m "docs(ter): F4 softmax done -- per-lane LUT pattern lessons"
```

---

## Self-Review

- **Spec coverage:** §8 kernel catalogue — `tk_softmax` covered by F4.7. `tk_swiglu`, `tk_rope`, `tk_attention` deferred.
- **Placeholder scan:** the kernel uses store-then-tvload broadcast and per-lane software lookup loops. Both are documented warts, not placeholders.
- **Type consistency:** `tk_softmax`, `gen_softmax_luts.py`, `exp_lut.bin`, `rcp_lut.bin` referenced consistently.
- **Known caveat:** no max-subtract → input must be in roughly `[-4, +4]` (the LUT's domain). The test uses `uniform(-2, 2)` to stay well within range. Real attention logits will overflow this and need stability handling. Documented.

---

## Execution Handoff

After all tasks complete, the project has:
- An exp LUT generator (256 entries, fixed-point int).
- A recip LUT generator (256 entries, for the normalisation).
- The first kernel doing per-lane LUT lookup via software loop.
- An end-to-end softmax test matching numpy within 10% rel_err.

Next plan: SwiGLU (sigmoid LUT, similar shape to softmax minus the recip).
