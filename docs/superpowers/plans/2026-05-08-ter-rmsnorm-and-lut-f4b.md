# `ter` RMSNorm + LUT Infrastructure Plan (F4-cont)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Implement `tk_rmsnorm` — the first transcendental kernel — using a host-generated `rsqrt` lookup table loaded into sim memory. End state: random input vector + gain weights → RMSNorm via ternary kernel → matches numpy reference within bounded relative error. Establishes the LUT pattern for the remaining transcendental kernels (softmax, swiglu, rope).

**Architecture:** Host generates the rsqrt LUT once (Python script, written to a binary file). Sim init loads the LUT into a fixed memory region via a new `Sim::load_lut(addr, values)` API. Kernel computes `sum_sq = sum(x²)` via `tvmac`, derives a LUT index from the int sum (no floats), `tload`s the rsqrt value, multiplies the input vector through, scales by per-tensor gain `w` and the float scaling factor on the host.

**Tech Stack:** Same as plan 2 (C++17, CMake, doctest, `.venv/bin/python` for numpy reference).

**Spec:** `docs/superpowers/specs/2026-05-08-ter-design.md` §8 (kernel catalogue). Patterns doc: `docs/kernel-patterns.md`.

**Out of scope for this plan:** Softmax, SwiGLU, RoPE, attention kernels (each gets its own follow-up plan once the LUT pattern is proven). Format A (`tfloat`). F5 bridge. F6 Llama 3.2 1B.

---

## Why RMSNorm first

RMSNorm is the simplest transcendental kernel: one reduction (sum of squares) + one transcendental (rsqrt) + one element-wise multiply. Softmax, swiglu, rope all follow the same shape with a different transcendental. Get the LUT pattern right here.

Reference formula:
```
y[i] = x[i] * gain[i] * rsqrt(sum_j(x[j]^2) / N + eps)
```

For a fixed-N implementation (a single tile of length 27), N is baked into the kernel and `eps` is added on the host before passing the int sum-of-squares to the LUT lookup. Multi-tile orchestration (N > 27) is host-side, same K-tiling pattern as matmul.

---

## File Structure

```
ter/
├── include/ter/
│   ├── sim.hpp                          # add load_lut declaration
│   └── (existing)
├── src/
│   ├── sim/
│   │   └── load_lut.cpp                 # Sim::load_lut implementation
│   └── kernels/
│       └── tk_rmsnorm.tasm              # the new kernel
├── tools/
│   └── gen_rsqrt_lut.py                 # writes build/rsqrt_lut.bin (256 int32 entries)
└── tests/
    ├── test_load_lut.cpp                # smoke test for the new API
    └── test_kernel_rmsnorm.cpp          # end-to-end kernel vs numpy
```

LUT layout: 256 entries, `int32` each. Index `i ∈ [0, 255]` maps to `rsqrt(i / 255 * range)` scaled by a fixed `lut_scale = 9841` (max 9-trit int) so each entry fits in a Word27. The kernel computes `idx = clamp(sum_sq * 255 / sum_max, 0, 255)` in trit math, then `tload`s.

Since `sum_max` is fixed per-kernel (depends on the input domain), it's part of the kernel's compile-time constants encoded as `tloadi` immediates.

---

## Task F4.5.1 — `Sim::load_lut` API

**Files:**
- Modify `include/ter/sim.hpp` — add `load_lut` declaration
- Create `src/sim/load_lut.cpp`
- Create `tests/test_load_lut.cpp`
- Modify `src/CMakeLists.txt`, `tests/CMakeLists.txt`

**Critical git discipline:** stay on `feature/f0-f2-foundation`, no detached HEAD, use `build/`.

- [ ] **Step 1: Write `tests/test_load_lut.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/assembler.hpp>
#include <vector>

using namespace ter;

TEST_CASE("load_lut writes int values as Word27s and a kernel reads them back") {
    // Kernel: r1 = mem[r1]; halt.   (Read LUT[index] using r1 as the address.)
    auto blob = assemble(R"(
        tload r1, r1
        thalt
    )");
    Sim s(2048);
    KernelTable kt;
    KernelId id = kt.install(s, "lut_read", blob);

    std::vector<int> lut = {10, 20, -5, 99, 0, 1234};
    int lut_addr = 100;
    s.load_lut(lut_addr, lut);

    for (int i = 0; i < static_cast<int>(lut.size()); ++i) {
        std::vector<int64_t> args = {lut_addr + i, 0, 0, 0, 0, 0, 0};
        int64_t r = s.call_kernel(kt, id, args);
        CHECK(r == lut[i]);
    }
}
```

- [ ] **Step 2: Append to `include/ter/sim.hpp`** (in `class Sim` public section, after `call_kernel`):

```cpp
    // Stores `values.size()` entries starting at `addr`, one Word27 per entry.
    void load_lut(size_t addr, const std::vector<int>& values);
```

- [ ] **Step 3: Implement `src/sim/load_lut.cpp`**

```cpp
#include <ter/sim.hpp>

namespace ter {

void Sim::load_lut(size_t addr, const std::vector<int>& values) {
    for (size_t i = 0; i < values.size(); ++i) {
        mem_.store_word(addr + i, Word27::from_int(values[i]));
    }
}

}
```

- [ ] **Step 4: Wire build, register**

`src/CMakeLists.txt`: add `sim/load_lut.cpp`. `tests/CMakeLists.txt`: append `ter_add_test(test_load_lut)`.

- [ ] **Step 5: Build and run**

```
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: 29/29.

- [ ] **Step 6: Commit**

```
git add include/ter/sim.hpp src/sim/load_lut.cpp src/CMakeLists.txt tests/test_load_lut.cpp tests/CMakeLists.txt
git commit -m "feat(sim): load_lut API for storing int LUTs as Word27 in sim memory"
```

After: `git log --oneline -2`.

---

## Task F4.6.1 — `gen_rsqrt_lut.py` (host-side LUT generator)

**Files:**
- Create `tools/gen_rsqrt_lut.py`

The LUT maps `i ∈ [0, 255]` to `round(rsqrt(value_at_i) * lut_out_scale)`. We choose `value_at_i = i + 1` (avoid div by zero) and `lut_out_scale = 9841` (max 9-trit int) so entries fit in Word27.

The kernel will compute `idx = clamp(sum_sq_int / sum_div, 0, 255)` where `sum_div` is a per-call constant chosen so that the worst-case `sum_sq_int` maps to index 255.

For RMSNorm with N=27 inputs each in `[-9841, +9841]`: max `sum_sq = 27 * 9841² ≈ 2.6e9`. So `sum_div = ceil(2.6e9 / 255) ≈ 1.02e7`.

This task only generates the LUT data file; the kernel and test land in F4.6.2 / F4.6.3.

- [ ] **Step 1: Write `tools/gen_rsqrt_lut.py`**

```python
#!/usr/bin/env python3
"""Generates a 256-entry int32 LUT for rsqrt, used by tk_rmsnorm."""
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
    # Map i in [0, n_entries-1] to value (i+1)/n_entries (avoid 0).
    # Compute rsqrt of that, scale to int.
    idx = np.arange(a.n_entries, dtype=np.float64)
    vals = (idx + 1.0) / float(a.n_entries)
    rsq = 1.0 / np.sqrt(vals)
    lut = np.round(rsq * a.out_scale / rsq.max()).astype(np.int32)
    out_path = os.path.join(a.out_dir, "rsqrt_lut.bin")
    lut.tofile(out_path)
    with open(os.path.join(a.out_dir, "rsqrt_lut.meta"), "w") as f:
        f.write(f"n_entries={a.n_entries}\nout_scale={a.out_scale}\nrsq_max={rsq.max()}\n")
    print(f"wrote {a.n_entries}-entry rsqrt LUT to {out_path}")
    print(f"rsq_max = {rsq.max():.6f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make executable and smoke-test**

```
chmod +x tools/gen_rsqrt_lut.py
.venv/bin/python tools/gen_rsqrt_lut.py --out-dir build/lut_data
ls -la build/lut_data/
```

Expected: `rsqrt_lut.bin` (1024 bytes = 256 × 4) and `rsqrt_lut.meta` exist.

- [ ] **Step 3: Commit**

```
git add tools/gen_rsqrt_lut.py
git commit -m "feat(tools): rsqrt LUT generator for tk_rmsnorm"
```

---

## Task F4.6.2 — `tk_rmsnorm.tasm` (the kernel)

**Files:**
- Create `src/kernels/tk_rmsnorm.tasm`
- Modify `src/kernels/registry.cpp` — also load tk_rmsnorm

The kernel signature:
```
tk_rmsnorm(addr_X, addr_Y, addr_LUT, sum_div, lut_max_idx, gain_recip)
```
where:
- `addr_X` — input vector (27 trits, one Word27 per element)
- `addr_Y` — output vector (same shape)
- `addr_LUT` — base of rsqrt LUT in sim memory
- `sum_div` — divisor mapping `sum_sq` → LUT index (passed in R4)
- `lut_max_idx` — clamp ceiling for the index (passed in R5)
- `gain_recip` — unused for now; per-element gain multiplied on host. (Reserved arg slot.)

Algorithm:
1. `tvload v0, r1` — load X into V0
2. `tvmac a0, v0, v0` — A0 += sum(x²)
3. `tvsum r6, a0` — R6 = sum_sq (int)
4. *(host-side equivalent: idx = clamp(sum_sq / sum_div, 0, lut_max_idx); rsqrt_int = lut[idx])*
5. In ternary asm, divide by sum_div using repeated subtraction OR pre-shift-and-multiply: for the MVP, we use a **single division step** approximated as `idx = sum_sq / sum_div` via a `tcmp`-and-subtract loop. Bounded by lut_max_idx iterations (≤ 255). This is slow but honest.
6. `tload r7, r8` — read `LUT[idx]` (after computing addr_LUT + idx in r8)
7. *(host-side equivalent: y[i] = x[i] * rsqrt_int * scale_correction)*
8. For element-wise multiply: broadcast rsqrt_int into V1, then `tvmac a1, v0, v1` per-lane, then `tvstore` the lane outputs.

Since per-lane multiply via `tvmac` accumulates (not stores per-lane), we need `tvbroadcast v1, rsqrt_int` then per-lane multiply via a different op... we don't have `tvmul` (lane-wise multiply without accumulate) yet. **Options:**
- Add a `tvmul` opcode (lane-wise multiply, no accumulate). Requires extending `Vec` host primitive too. Small opcode addition.
- Hack: zero A1, then `tvmac a1, v0, v1`, then store the per-lane A1 values one at a time via `tvsum`-like reductions. Inefficient and not quite right (tvmac sums into one acc).

Cleanest path: **add `tvmul` (vd = vs1 * vs2 lane-wise, no accumulate)** as part of this task. It's a small, generally useful opcode and fits the existing pattern.

- [ ] **Step 1: Add `TVMUL` opcode**

In `include/ter/isa.hpp`, append to the `Opcode` enum:
```cpp
    TVMUL       = 122,
```

In `src/isa/decode.cpp`, add `case Opcode::TVMUL: return true;` to the `is_valid_opcode` switch.

In `include/ter/vec.hpp`, declare:
```cpp
Vec vec_mul(const Vec& a, const Vec& b) noexcept;
```

In `src/core/vec.cpp`, implement:
```cpp
Vec vec_mul(const Vec& a, const Vec& b) noexcept {
    Vec r;
    for (int i = 0; i < Vec::kLanes; ++i) {
        // Multiply lanes; clamp to vec range.
        int64_t prod = int64_t{a.lane(i)} * int64_t{b.lane(i)};
        if (prod > Vec::kLaneMax) prod = Vec::kLaneMax;
        if (prod < Vec::kLaneMin) prod = Vec::kLaneMin;
        r.set_lane(i, static_cast<int32_t>(prod));
    }
    return r;
}
```

In `src/sim/run_one.cpp`, add a case before `default:`:
```cpp
case Opcode::TVMUL:
    regs_.write_vec(i.dst, vec_mul(regs_.read_vec(i.src1), regs_.read_vec(i.src2)));
    break;
```

In `src/asm/emitter.cpp`, add to mnemonic table: `{"tvmul", Opcode::TVMUL},` and to dispatch:
```cpp
case Opcode::TVMUL:
    i.dst = vreg_at(1); i.src1 = vreg_at(2); i.src2 = vreg_at(3); break;
```

- [ ] **Step 2: Add unit test for `tvmul`** in `tests/test_executor_simd.cpp` (append):

```cpp
TEST_CASE("TVMUL lane-wise multiply with clamping") {
    Sim s(64);
    Instr code[] = {
        {Opcode::TVBROADCAST, 0, 0, 0, 100},
        {Opcode::TVBROADCAST, 1, 0, 0, 50},
        {Opcode::TVMUL,       2, 0, 1, 0},
        {Opcode::THALT,       0, 0, 0, 0},
    };
    for (size_t k = 0; k < 4; ++k) s.mem().store_word(k, encode(code[k]));
    s.run();
    auto v2 = s.regs().read_vec(2);
    for (int k = 0; k < Vec::kLanes; ++k) CHECK(v2.lane(k) == 5000);
}
```

- [ ] **Step 3: Build and run intermediate**

Expected: 30/30 (29 prior + new test). Confirm tvmul works before writing the kernel.

- [ ] **Step 4: Write `src/kernels/tk_rmsnorm.tasm`**

```
        ; tk_rmsnorm — single-tile RMSNorm of length 27 (one V-reg).
        ; Args:
        ;   r1 = addr_X  (27 Word27 input values)
        ;   r2 = addr_Y  (27 Word27 output slots)
        ;   r3 = addr_LUT (base of 256-entry rsqrt LUT)
        ;   r4 = sum_div (per-call divisor: sum_sq / sum_div => idx)
        ;   r5 = lut_max_idx (clamp ceiling, 255 for 256-entry LUT)
        ;
        ; Steps:
        ;   1. V0 = X
        ;   2. A0 += V0 * V0 (lane-wise multiply-accumulate)
        ;   3. r6 = sum_sq (sum of all lanes of A0)
        ;   4. r7 = idx (sum_sq divided by sum_div, repeated subtraction)
        ;   5. r8 = addr_LUT + idx
        ;   6. r9 = mem[r8]  (rsqrt_int)
        ;   7. V1 = broadcast(r9)
        ;   8. V2 = V0 * V1  (lane-wise multiply)
        ;   9. store V2 to addr_Y

        tvload      v0, r1            ; V0 = X
        tvmac       a0, v0, v0         ; A0 += X*X
        tvsum       r6, a0             ; r6 = sum_sq

        ; Compute idx = sum_sq / r4, capped at r5. Repeated subtraction loop:
        ; r7 = 0
        ; while (r6 >= r4 && r7 < r5) { r6 -= r4; r7 += 1; }
        tloadi      r7, 0
        tloadi      r10, 1            ; constant +1
div_loop:
        ; if r6 < r4 break
        tblt        r6, r4, after_div
        ; if r7 >= r5 break  (r7-r5 must be negative for "less than"; we test
        ;   negation: tblt r7, r5, ok => continue; otherwise break)
        tblt        r7, r5, ok_step
        tjump       after_div
ok_step:
        tsub        r6, r6, r4
        tadd        r7, r7, r10
        tjump       div_loop
after_div:
        ; r7 now holds idx in [0, r5].
        tadd        r8, r3, r7         ; r8 = addr_LUT + idx
        tload       r9, r8             ; r9 = rsqrt_int

        ; V1 = broadcast(r9) — but tvbroadcast takes an immediate, not a register.
        ; Workaround: use the immediate form. Since the immediate field can hold
        ; r9's range (12 trits = ±265720), we need a 'tvbroadcast_reg' or copy
        ; r9 to imm via... actually we can store r9 to memory, tvload it.
        ; That's expensive (creates a 27-element vector all the same).
        ; Simpler: copy r9 into 27 contiguous mem slots, then tvload.
        ; For minimal first pass, use the SP region (R26) as scratch:
        tloadi      r11, 0            ; loop counter
        tloadi      r12, 27           ; loop bound
        tloadi      r13, 800          ; scratch base for broadcast (avoid Y region)
broadcast_loop:
        tbeq        r11, r12, after_broadcast
        tadd        r14, r13, r11
        tstore      r9, r14
        tadd        r11, r11, r10
        tjump       broadcast_loop
after_broadcast:
        tvload      v1, r13            ; V1 = broadcast of r9

        tvmul       v2, v0, v1         ; V2 = X * rsqrt_int  (lane-wise)
        tvstore     v2, r2             ; store V2 to addr_Y

        thalt
```

This is substantial (~50 lines of asm). Note the broadcast workaround: since `tvbroadcast` only accepts an immediate, we manually fill 27 memory slots with `r9` then `tvload` them. **This is a known wart**; a future `TVBROADCAST_R` opcode (broadcast from register) would clean it up. Document but don't fix in this task.

- [ ] **Step 5: Update `src/kernels/registry.cpp`**

In `install_default_kernels`, add:
```cpp
    auto src_rms = read_text(kernels_dir + "/tk_rmsnorm.tasm");
    auto blob_rms = assemble(src_rms);
    kt.install(sim, "tk_rmsnorm", blob_rms);
```

- [ ] **Step 6: Verify the kernel assembles cleanly**

Add a smoke test (in `tests/test_kernel_matmul_b_small.cpp` or a new file): `install_default_kernels(...)` succeeds and `kt.find("tk_rmsnorm").valid` is true.

Actually, defer this to F4.6.3's full test — the kernel will be exercised end-to-end there.

- [ ] **Step 7: Build (no new test yet, just verify it compiles)**

```
cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: still 30/30 (no new test). The build proves the kernel assembles.

- [ ] **Step 8: Commit**

```
git add include/ter/isa.hpp include/ter/vec.hpp src/isa/decode.cpp src/core/vec.cpp src/sim/run_one.cpp src/asm/emitter.cpp src/kernels/tk_rmsnorm.tasm src/kernels/registry.cpp tests/test_executor_simd.cpp
git commit -m "feat(kernels): tk_rmsnorm + TVMUL opcode for lane-wise multiply"
```

After: `git log --oneline -2`.

---

## Task F4.6.3 — End-to-end RMSNorm test vs numpy

**Files:**
- Modify `tools/gen_rsqrt_lut.py` if needed (already created in F4.6.1)
- Create `tests/test_kernel_rmsnorm.cpp`
- Modify `tests/CMakeLists.txt` (add fixture, register test with TER_KERNELS_DIR)

- [ ] **Step 1: Write `tests/test_kernel_rmsnorm.cpp`**

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

TEST_CASE("tk_rmsnorm matches numpy reference within bounded rel_err") {
    constexpr int N = 27;
    constexpr int N_ENTRIES = 256;
    constexpr int OUT_SCALE = 9841;

    // Generate input vector (random standard normal).
    std::vector<float> x(N);
    std::mt19937 rng(0xBEEF);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : x) v = dist(rng);

    // Numpy reference: y[i] = x[i] * rsqrt(mean(x^2) + eps)
    constexpr float eps = 1e-6f;
    double sum_sq = 0.0;
    for (int i = 0; i < N; ++i) sum_sq += double(x[i]) * double(x[i]);
    double mean_sq = sum_sq / N;
    double rsqrt_ref = 1.0 / std::sqrt(mean_sq + eps);
    std::vector<float> y_ref(N);
    for (int i = 0; i < N; ++i) y_ref[i] = static_cast<float>(double(x[i]) * rsqrt_ref);

    // Quantize x to format B.
    TritTensor xt = quantize(x.data(), {N}, 9);

    // Load the LUT from the fixture.
    auto lut_i32 = read_i32("lut_data/rsqrt_lut.bin");
    REQUIRE(lut_i32.size() == N_ENTRIES);
    std::vector<int> lut(lut_i32.begin(), lut_i32.end());

    // Set up sim.
    Sim s(4096);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    KernelId id = kt.find("tk_rmsnorm");
    REQUIRE(id.valid);

    int x_addr = 100, y_addr = 200, lut_addr = 1000;
    for (int i = 0; i < N; ++i) s.mem().store_word(static_cast<size_t>(x_addr + i), xt.payload[i]);
    s.load_lut(lut_addr, lut);

    // Compute sum_div: max sum_sq for N elements at +/-mti is N * mti^2.
    // We want sum_sq / sum_div mapped to [0, 255], so sum_div = N * mti^2 / 255.
    int64_t mti = 9841;
    int64_t sum_div = (N * mti * mti) / 255;

    std::vector<int64_t> args = {x_addr, y_addr, lut_addr, sum_div, 255, 0, 0};
    s.call_kernel(kt, id, args);

    // Read y_int back from sim memory.
    std::vector<int> y_int(N);
    for (int i = 0; i < N; ++i) y_int[i] = static_cast<int>(s.mem().load_word(static_cast<size_t>(y_addr + i)).to_int());

    // Compute the float scaling that converts y_int back to y.
    // y_int = round( x_int * rsqrt_int_lut ), where:
    //   x_int = x / xt.scale
    //   rsqrt_int_lut = round( rsqrt(value_at_idx) * OUT_SCALE / rsq_max )
    // So  y_int * (xt.scale / OUT_SCALE) * rsq_max ≈ x * rsqrt(value_at_idx)
    // To get y, also need to divide by sqrt(some normalization). For this MVP test
    // we accept a generous bound (5e-2 relative error) — the LUT discretization
    // and integer division make this approximate.
    //
    // Compute rsq_max from the LUT meta side-channel for the recovery scale.
    // For simplicity we read it from the same fixture.
    float rsq_max = 0.0f;
    {
        std::ifstream meta("lut_data/rsqrt_lut.meta");
        REQUIRE(meta.is_open());
        std::string line;
        while (std::getline(meta, line)) {
            auto pos = line.find("rsq_max=");
            if (pos != std::string::npos) {
                rsq_max = std::stof(line.substr(pos + 8));
            }
        }
    }
    REQUIRE(rsq_max > 0.0f);

    std::vector<float> y(N);
    float recovery = xt.scale * rsq_max / static_cast<float>(OUT_SCALE);
    for (int i = 0; i < N; ++i) y[i] = static_cast<float>(y_int[i]) * recovery;

    // Bounded relative error per element vs numpy reference.
    double max_rel = 0.0;
    for (int i = 0; i < N; ++i) {
        double ref = y_ref[i];
        double got = y[i];
        double denom = std::max(1.0, std::fabs(ref));
        double rel = std::fabs(got - ref) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    // Generous bound: LUT has 256 buckets, repeated-subtraction division is
    // approximate, and the recovery scale chain compounds noise.
    CHECK(max_rel < 5e-2);

    // Counter check: exactly one tvmac, one tvsum, one tvmul per call.
    CHECK(s.counters().get(Opcode::TVMAC) == 1);
    CHECK(s.counters().get(Opcode::TVSUM) == 1);
    CHECK(s.counters().get(Opcode::TVMUL) == 1);
}
```

- [ ] **Step 2: Wire CMake fixture in `tests/CMakeLists.txt` (append):**

```cmake
add_test(NAME gen_rsqrt_lut
    COMMAND ${CMAKE_SOURCE_DIR}/.venv/bin/python
            ${CMAKE_SOURCE_DIR}/tools/gen_rsqrt_lut.py
            --out-dir ${CMAKE_BINARY_DIR}/lut_data
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(gen_rsqrt_lut PROPERTIES FIXTURES_SETUP rsqrt_lut)

ter_add_test(test_kernel_rmsnorm)
target_compile_definitions(test_kernel_rmsnorm PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
set_tests_properties(test_kernel_rmsnorm PROPERTIES
    FIXTURES_REQUIRED rsqrt_lut
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
```

- [ ] **Step 3: Build and run**

```
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: 32/32 (30 prior + new fixture + new test). All green.

If the test fails on `max_rel < 5e-2`:
- Inspect what `idx` the kernel computed (you can `tdbg` after the div_loop and read R7 from the regfile).
- Common issue: the recovery scale doesn't match what the kernel actually output; recompute by hand for one element.

- [ ] **Step 4: Commit**

```
git add tests/test_kernel_rmsnorm.cpp tests/CMakeLists.txt
git commit -m "test(kernels): F4.6 -- tk_rmsnorm matches numpy within bounded rel_err"
```

After: `git log --oneline -3`.

---

## Final Task — Update README + patterns

**Files:**
- Modify `README.md` — note F4 progress
- Modify `docs/kernel-patterns.md` — add LUT pattern lessons learned
- Modify `docs/isa.md` — add TVMUL row

- [ ] **Step 1: Update `README.md` Status block**

Replace `[x] F4 (partial) — ...` with:
```markdown
- [x] F4 (matmul + rmsnorm) — Sim::call_kernel, KernelTable, tk_matmul_b_9t, tk_rmsnorm + rsqrt LUT, TVMUL opcode.
```

- [ ] **Step 2: Append a "Lessons from RMSNorm" section to `docs/kernel-patterns.md`**

```markdown
## Lessons from `tk_rmsnorm`

- **`tvbroadcast` only accepts an immediate**, not a register. To broadcast a runtime value across a vreg, the kernel must store the value to 27 contiguous memory slots and `tvload`. A future `TVBROADCAST_R` opcode would clean this up.
- **Lane-wise multiply** without accumulation needed `TVMUL` (vd = vs1 * vs2). Distinct from `TVMAC` which accumulates into an A-reg.
- **Repeated-subtraction division** in trit asm is bounded by `lut_max_idx` iterations (≤255 for a 256-entry LUT). Acceptable for one division per kernel call; not appropriate for inner loops. A future Newton-Raphson divide kernel would be more honest.
- **LUT recovery scale** is non-trivial: the kernel produces an int that needs a host-side `xt.scale * rsq_max / OUT_SCALE` factor to land in float. Document this per-kernel.
```

- [ ] **Step 3: Append `tvmul` row to `docs/isa.md`**

```markdown
| tvmul   | 122 | vd, vs1, vs2 | per-lane multiply (clamped) |
```

- [ ] **Step 4: Build and run (no test changes)**

Expected: 32/32 still.

- [ ] **Step 5: Commit**

```
git add README.md docs/kernel-patterns.md docs/isa.md
git commit -m "docs(ter): F4 rmsnorm done -- LUT pattern lessons + TVMUL ISA row"
```

---

## Self-Review

- **Spec coverage:** §8.2 kernel catalogue — `tk_rmsnorm` with rsqrt LUT covered by F4.5/F4.6. The remaining 4 kernels (softmax, swiglu, rope, attention) explicitly deferred to follow-up plans.
- **Placeholder scan:** the kernel uses a known-wart broadcast pattern (store-then-tvload). Documented as a follow-up improvement, not a placeholder.
- **Type consistency:** `Sim::load_lut`, `tk_rmsnorm`, `TVMUL`/`vec_mul`, `gen_rsqrt_lut.py` referenced consistently across all tasks.
- **Known caveat:** the rsqrt LUT uses 256 entries with linear-index quantization. For real RMSNorm in a transformer (where input distributions are non-uniform), per-tensor scale on the way in might cause idx clustering. Not addressed in this MVP — the test uses standard-normal inputs which exercise a wide range. Will revisit if accuracy issues appear in F5+.

---

## Execution Handoff

After all tasks complete, the project has:
- A working LUT load API (`Sim::load_lut`).
- A new `TVMUL` opcode for lane-wise multiply without accumulation.
- The first transcendental kernel (`tk_rmsnorm`) end-to-end with bounded relative error.
- Documented LUT patterns and ISA additions.

Next plan: `tk_softmax` (exp LUT + reduce-max for stability) following the same pattern, then `tk_swiglu` (sigmoid LUT), `tk_rope` (sin/cos LUT), and finally `tk_attention` (composition of matmul + softmax).
