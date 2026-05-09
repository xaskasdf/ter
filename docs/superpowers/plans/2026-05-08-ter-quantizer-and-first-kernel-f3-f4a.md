# `ter` Quantizer + First K3 Kernel Plan (Phases F3 + F4-partial)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the format-B quantizer (bf16/float → fixed-point ternary with per-tensor scale and back) plus the first K3 ternary bytecode kernel — `tk_matmul_b_9t` — invoked from the host via `Sim::call_kernel`. End state: a kernel-program matmul executes inside the simulator on quantized inputs, producing outputs that match numpy reference within bounded relative error, with operation counts reported.

**Architecture:** Quantizer is host-side C++ producing packed trit payloads + scale metadata. Kernel is a `.tasm` file assembled at sim init into the simulator's code segment. Host `call_kernel` pushes arguments to R1..R7 and runs the sim from the kernel entry until `thalt`. The matmul kernel uses `tvmac` over 27-lane chunks and `tvsum` to reduce, accumulating in A0.

**Tech Stack:** C++17, CMake, doctest, Python 3 + numpy (already wired in F2.4 via `.venv/`).

**Spec:** `docs/superpowers/specs/2026-05-08-ter-design.md` — §7.1 (Format B), §8 (K3 kernels)

**Out of scope for this plan:** The remaining F4 kernels (rmsnorm, rope, swiglu, softmax, attention) — own follow-up plan. F5 (ntransformer bridge), F6 (Llama 3.2 1B end-to-end), F7-F10 optionals. Format A (`tfloat`) is also deferred.

---

## File Structure (created during this plan)

```
ter/
├── include/ter/
│   ├── numfmt.hpp                       # DType enum, TritTensor struct, quantize/dequantize
│   └── kernels.hpp                      # KernelId, KernelTable, Sim::call_kernel signature
├── src/
│   ├── numfmt/
│   │   ├── quantize.cpp                 # float[] → packed trits + scale
│   │   ├── dequantize.cpp               # packed trits + scale → float[]
│   │   └── tensor.cpp                   # TritTensor: load into sim memory
│   ├── kernels/
│   │   ├── tk_matmul_b_9t.tasm          # the matmul kernel source
│   │   └── registry.cpp                 # assembles .tasm at init, registers in KernelTable
│   └── sim/
│       └── call_kernel.cpp              # Sim::call_kernel: push args, jump, run
└── tests/
    ├── test_numfmt_quantize.cpp         # quantization basics
    ├── test_numfmt_roundtrip.cpp        # bounded MSE round-trip (fuzz)
    ├── test_kernel_matmul_b_small.cpp   # 9x9x9 matmul via kernel vs numpy
    └── test_kernel_matmul_b_32.cpp      # 32x32x32 matmul via kernel, bounded rel_err vs numpy
```

The matmul kernel is non-trivial (loops + tvmac chunks + scaling on host). It is the first proof that the K3 invocation pattern works end-to-end. Subsequent kernels (in the next plan) reuse the registry and call_kernel infrastructure.

---

## Task F3.1 — DType enum and `TritTensor` metadata

**Files:**
- Create: `include/ter/numfmt.hpp`
- Create: `tests/test_numfmt_metadata.cpp`
- Modify: `tests/CMakeLists.txt`

**Critical git discipline:** Stay on `feature/f0-f2-foundation` (or a successor branch — verify with `git status` first). DO NOT detach HEAD. Use the existing `build/` directory.

- [ ] **Step 1: Write `tests/test_numfmt_metadata.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/numfmt.hpp>

using namespace ter;

TEST_CASE("DType enum values are stable") {
    CHECK(static_cast<int>(DType::Float32) == 0);
    CHECK(static_cast<int>(DType::TritFP_B) == 1);
}

TEST_CASE("TritTensor default-constructible and inspectable") {
    TritTensor t;
    CHECK(t.dtype == DType::TritFP_B);
    CHECK(t.n_trits_per_elem == 9);
    CHECK(t.scale == 0.0f);
    CHECK(t.shape.empty());
    CHECK(t.payload.empty());
}

TEST_CASE("TritTensor can be sized and shaped") {
    TritTensor t;
    t.shape = {4, 8};
    t.n_trits_per_elem = 9;
    t.payload.resize(4 * 8);  // one Word27 per element (each holds n_trits_per_elem trits)
    CHECK(t.num_elems() == 32);
    CHECK(t.payload.size() == 32);
}
```

- [ ] **Step 2: Write `include/ter/numfmt.hpp`**

```cpp
#pragma once
#include <ter/word.hpp>
#include <vector>
#include <cstdint>

namespace ter {

enum class DType : int {
    Float32  = 0,
    TritFP_B = 1,
};

// A tensor in format B: integer ternary payload + per-tensor float32 scale.
// Each element occupies one Word27 in `payload` (lower n_trits_per_elem trits valid).
struct TritTensor {
    DType dtype = DType::TritFP_B;
    int   n_trits_per_elem = 9;            // default: 9 trits per element (~int10)
    float scale = 0.0f;                    // payload_int * scale = original float value
    std::vector<int> shape;                // dimensions, row-major
    std::vector<Word27> payload;           // num_elems() entries

    size_t num_elems() const noexcept {
        if (shape.empty()) return 0;
        size_t n = 1;
        for (int d : shape) n *= static_cast<size_t>(d);
        return n;
    }
};

// Largest absolute integer representable in n trits (balanced).
constexpr int64_t max_trit_int(int n_trits) noexcept {
    int64_t v = 1;
    for (int i = 0; i < n_trits; ++i) v *= 3;
    return (v - 1) / 2;
}

}  // namespace ter
```

- [ ] **Step 3: Wire build, register test**

In `tests/CMakeLists.txt` append `ter_add_test(test_numfmt_metadata)`.

- [ ] **Step 4: Build and run**

```
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: 22/22 (21 prior + new test).

- [ ] **Step 5: Commit**

```
git add include/ter/numfmt.hpp tests/test_numfmt_metadata.cpp tests/CMakeLists.txt
git commit -m "feat(numfmt): DType enum and TritTensor metadata"
```

---

## Task F3.2 — Quantize float → format B (per-tensor scale)

**Files:**
- Modify: `include/ter/numfmt.hpp` — add `quantize` declaration
- Create: `src/numfmt/quantize.cpp`
- Create: `tests/test_numfmt_quantize.cpp`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`

- [ ] **Step 1: Write `tests/test_numfmt_quantize.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/numfmt.hpp>
#include <vector>
#include <cmath>

using namespace ter;

TEST_CASE("quantize a small float vector with positive max") {
    std::vector<float> xs = {0.1f, -0.5f, 0.25f, 1.0f};
    TritTensor t = quantize(xs.data(), {4}, /*n_trits=*/9);
    CHECK(t.shape == std::vector<int>{4});
    CHECK(t.n_trits_per_elem == 9);
    // max abs = 1.0, so scale = 1.0 / max_trit_int(9) = 1/9841
    CHECK(t.scale == doctest::Approx(1.0f / 9841.0f));
    REQUIRE(t.payload.size() == 4);
    // Spot-check: 1.0f / scale ≈ 9841 ≈ all-positive trits
    int64_t last = t.payload[3].to_int();
    CHECK(last >= 9000);
    CHECK(last <= 9841);
}

TEST_CASE("quantize a vector of zeros yields zero scale") {
    std::vector<float> xs = {0.0f, 0.0f, 0.0f};
    TritTensor t = quantize(xs.data(), {3}, 9);
    // All-zero tensor: scale defined as 0.0 (not NaN), payload all-zero.
    CHECK(t.scale == 0.0f);
    for (auto& w : t.payload) CHECK(w.to_int() == 0);
}

TEST_CASE("quantize handles negative-only input") {
    std::vector<float> xs = {-0.5f, -1.0f, -0.25f};
    TritTensor t = quantize(xs.data(), {3}, 9);
    CHECK(t.scale > 0.0f);
    CHECK(t.payload[0].to_int() < 0);
    CHECK(t.payload[1].to_int() < 0);
}
```

- [ ] **Step 2: Append to `include/ter/numfmt.hpp`**

```cpp
// Quantize a flat float buffer to format B with per-tensor scale.
// shape: dimensions, must product to nelems. n_trits_per_elem: 9 default.
TritTensor quantize(const float* data,
                    const std::vector<int>& shape,
                    int n_trits_per_elem = 9);
```

- [ ] **Step 3: Implement `src/numfmt/quantize.cpp`**

```cpp
#include <ter/numfmt.hpp>
#include <cmath>
#include <stdexcept>

namespace ter {

TritTensor quantize(const float* data, const std::vector<int>& shape, int n_trits_per_elem) {
    if (n_trits_per_elem < 1 || n_trits_per_elem > Word27::kTrits) {
        throw std::out_of_range("quantize: n_trits_per_elem must be in [1, 27]");
    }
    TritTensor t;
    t.dtype = DType::TritFP_B;
    t.n_trits_per_elem = n_trits_per_elem;
    t.shape = shape;

    size_t n = t.num_elems();
    t.payload.resize(n);

    // Find max absolute value.
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float a = std::fabs(data[i]);
        if (a > max_abs) max_abs = a;
    }

    int64_t mti = max_trit_int(n_trits_per_elem);
    if (max_abs == 0.0f) {
        t.scale = 0.0f;
        // payload already zero-initialised via Word27{}
        return t;
    }

    t.scale = max_abs / static_cast<float>(mti);

    // Quantize each element: round(value / scale) clamped to [-mti, +mti].
    for (size_t i = 0; i < n; ++i) {
        float q = data[i] / t.scale;
        int64_t r = static_cast<int64_t>(std::lround(q));
        if (r > mti)  r = mti;
        if (r < -mti) r = -mti;
        t.payload[i] = Word27::from_int(r);
    }
    return t;
}

}  // namespace ter
```

- [ ] **Step 4: Wire build**

In `src/CMakeLists.txt` add `numfmt/quantize.cpp` to the source list.
In `tests/CMakeLists.txt` append `ter_add_test(test_numfmt_quantize)`.

- [ ] **Step 5: Build and run**

```
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: 23/23.

- [ ] **Step 6: Commit**

```
git add include/ter/numfmt.hpp src/numfmt src/CMakeLists.txt tests/test_numfmt_quantize.cpp tests/CMakeLists.txt
git commit -m "feat(numfmt): quantize float to format B with per-tensor scale"
```

---

## Task F3.3 — Dequantize format B → float

**Files:**
- Modify: `include/ter/numfmt.hpp` — add `dequantize` declaration
- Create: `src/numfmt/dequantize.cpp`
- Modify: `src/CMakeLists.txt`
- Append to `tests/test_numfmt_quantize.cpp`

- [ ] **Step 1: Append to `tests/test_numfmt_quantize.cpp`**

```cpp
TEST_CASE("dequantize is approximate inverse of quantize") {
    std::vector<float> xs = {0.1f, -0.5f, 0.25f, 1.0f, -0.75f};
    TritTensor t = quantize(xs.data(), {5}, 9);
    std::vector<float> ys(5);
    dequantize(t, ys.data());
    // Expected error: at most scale/2 per element (round-to-nearest).
    for (size_t i = 0; i < xs.size(); ++i) {
        CHECK(std::fabs(ys[i] - xs[i]) <= t.scale * 0.5f + 1e-6f);
    }
}

TEST_CASE("dequantize zero tensor returns zeros") {
    std::vector<float> xs = {0.0f, 0.0f};
    TritTensor t = quantize(xs.data(), {2}, 9);
    std::vector<float> ys(2);
    dequantize(t, ys.data());
    CHECK(ys[0] == 0.0f);
    CHECK(ys[1] == 0.0f);
}
```

- [ ] **Step 2: Append to `include/ter/numfmt.hpp`**

```cpp
// Dequantize: out[i] = payload[i].to_int() * scale.
void dequantize(const TritTensor& t, float* out);
```

- [ ] **Step 3: Implement `src/numfmt/dequantize.cpp`**

```cpp
#include <ter/numfmt.hpp>

namespace ter {

void dequantize(const TritTensor& t, float* out) {
    size_t n = t.num_elems();
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(t.payload[i].to_int()) * t.scale;
    }
}

}  // namespace ter
```

- [ ] **Step 4: Wire build**

Add `numfmt/dequantize.cpp` to `src/CMakeLists.txt`.

- [ ] **Step 5: Build and run**

Expected: 23/23 (existing test count grew internally).

- [ ] **Step 6: Commit**

```
git add include/ter/numfmt.hpp src/numfmt/dequantize.cpp src/CMakeLists.txt tests/test_numfmt_quantize.cpp
git commit -m "feat(numfmt): dequantize format B to float"
```

---

## Task F3.4 — Round-trip MSE under bound (fuzz)

**Files:**
- Create: `tests/test_numfmt_roundtrip.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write `tests/test_numfmt_roundtrip.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/numfmt.hpp>
#include <random>
#include <cmath>
#include <vector>

using namespace ter;

static double mse(const std::vector<float>& a, const std::vector<float>& b) {
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = double(a[i]) - double(b[i]);
        acc += d * d;
    }
    return acc / static_cast<double>(a.size());
}

TEST_CASE("round-trip MSE under quantization-noise bound (uniform [-1,1])") {
    constexpr int N = 4096;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> xs(N);
    for (auto& v : xs) v = u(rng);

    TritTensor t = quantize(xs.data(), {N}, 9);
    std::vector<float> ys(N);
    dequantize(t, ys.data());

    // Quantization noise: round-to-nearest with step = scale.
    // Variance per element is at most scale^2 / 12 (uniform error in [-scale/2, scale/2]).
    double bound = static_cast<double>(t.scale) * static_cast<double>(t.scale) / 12.0 * 4.0;
    // Multiply by 4 to leave headroom for numerical/edge effects; the test catches
    // gross errors (factor of 100+), not borderline drift.
    CHECK(mse(xs, ys) < bound);
}

TEST_CASE("round-trip MSE drops as n_trits increases") {
    constexpr int N = 1024;
    std::mt19937 rng(67890);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> xs(N);
    for (auto& v : xs) v = u(rng);

    std::vector<float> ys(N);
    TritTensor t9  = quantize(xs.data(), {N}, 9);
    dequantize(t9, ys.data());
    double m9 = mse(xs, ys);

    TritTensor t12 = quantize(xs.data(), {N}, 12);
    dequantize(t12, ys.data());
    double m12 = mse(xs, ys);

    // Each extra trit divides the step by 3, so MSE divides by 9.
    // Allow generous slack — just confirm that more trits = less error.
    CHECK(m12 < m9 / 4.0);
}
```

- [ ] **Step 2: Register test, build, run**

`tests/CMakeLists.txt`: append `ter_add_test(test_numfmt_roundtrip)`.

Expected: 24/24.

- [ ] **Step 3: Commit**

```
git add tests/test_numfmt_roundtrip.cpp tests/CMakeLists.txt
git commit -m "test(numfmt): round-trip MSE bounds + monotone with n_trits"
```

---

## Task F4.1 — `Sim::call_kernel` and `KernelTable`

**Files:**
- Create: `include/ter/kernels.hpp`
- Modify: `include/ter/sim.hpp` — add `call_kernel`, `KernelTable&`
- Create: `src/sim/call_kernel.cpp`
- Create: `tests/test_call_kernel.cpp`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`

- [ ] **Step 1: Write `tests/test_call_kernel.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/assembler.hpp>

using namespace ter;

TEST_CASE("call_kernel runs assembled program with args in R1..R7") {
    // Tiny kernel: r1 = r1 + r2; halt.
    // (Result ends up in R1 which is also the first return register by convention.)
    auto blob = assemble(R"(
        tadd r1, r1, r2
        thalt
    )");
    Sim s(256);
    KernelTable kt;
    KernelId id = kt.install(s, "test_add", blob);

    std::vector<int64_t> args = {10, 32, 0, 0, 0, 0, 0};   // r1=10, r2=32, rest=0
    int64_t r1 = s.call_kernel(kt, id, args);
    CHECK(r1 == 42);
}

TEST_CASE("call_kernel preserves OpCounters across invocations") {
    auto blob = assemble(R"(
        tadd r1, r1, r2
        thalt
    )");
    Sim s(256);
    KernelTable kt;
    KernelId id = kt.install(s, "addk", blob);

    std::vector<int64_t> a1 = {1, 1, 0, 0, 0, 0, 0};
    s.call_kernel(kt, id, a1);
    s.call_kernel(kt, id, a1);

    CHECK(s.counters().get(Opcode::TADD)  == 2);
    CHECK(s.counters().get(Opcode::THALT) == 2);
}
```

- [ ] **Step 2: Write `include/ter/kernels.hpp`**

```cpp
#pragma once
#include <ter/word.hpp>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ter {

class Sim;

// Identifier returned at install time; opaque entry-address into sim memory.
struct KernelId {
    size_t entry_addr = 0;
    bool valid = false;
};

class KernelTable {
public:
    // Loads a kernel blob into the next free address of sim memory and registers it.
    KernelId install(Sim& sim, const std::string& name, const std::vector<Word27>& blob);

    KernelId find(const std::string& name) const noexcept;

private:
    std::unordered_map<std::string, KernelId> by_name_;
    size_t next_addr_ = 0;   // simple bump allocator for kernel code
};

}  // namespace ter
```

- [ ] **Step 3: Modify `include/ter/sim.hpp`**

Add `#include <vector>` and `#include <cstdint>` at the top.

Inside `class Sim` public section, after the existing `void run();` line, add:
```cpp
    // Sets up registers from `args` (R1..R7), jumps to kernel entry, runs until thalt.
    // Returns final R1 value (convention: first return register).
    int64_t call_kernel(class KernelTable& kt, struct KernelId id,
                        const std::vector<int64_t>& args);
```

- [ ] **Step 4: Write `src/sim/call_kernel.cpp`**

```cpp
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <stdexcept>

namespace ter {

KernelId KernelTable::install(Sim& sim, const std::string& name,
                              const std::vector<Word27>& blob) {
    KernelId id;
    id.entry_addr = next_addr_;
    id.valid = true;
    for (size_t i = 0; i < blob.size(); ++i) {
        sim.mem().store_word(next_addr_++, blob[i]);
    }
    by_name_[name] = id;
    return id;
}

KernelId KernelTable::find(const std::string& name) const noexcept {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? KernelId{} : it->second;
}

int64_t Sim::call_kernel(KernelTable&, KernelId id,
                         const std::vector<int64_t>& args) {
    if (!id.valid) throw std::runtime_error("call_kernel: invalid KernelId");
    if (args.size() > 7) throw std::runtime_error("call_kernel: max 7 register args");

    for (size_t i = 0; i < args.size(); ++i) {
        regs_.write_scalar(static_cast<int>(i + 1), Word27::from_int(args[i]));
    }
    regs_.set_pc(Word27::from_int(static_cast<int64_t>(id.entry_addr)));
    regs_.set_halted(false);
    run();
    return regs_.read_scalar(1).to_int();
}

}  // namespace ter
```

- [ ] **Step 5: Wire build**

In `src/CMakeLists.txt` add `sim/call_kernel.cpp`.

In `tests/CMakeLists.txt` append `ter_add_test(test_call_kernel)`.

- [ ] **Step 6: Build and run**

Expected: 25/25.

- [ ] **Step 7: Commit**

```
git add include/ter/kernels.hpp include/ter/sim.hpp src/sim/call_kernel.cpp src/CMakeLists.txt tests/test_call_kernel.cpp tests/CMakeLists.txt
git commit -m "feat(sim): KernelTable + call_kernel for K3 host-driven invocation"
```

---

## Task F4.2 — `tk_matmul_b_9t.tasm` — the matmul kernel in ternary asm

**Files:**
- Create: `src/kernels/tk_matmul_b_9t.tasm`
- Create: `src/kernels/registry.cpp` — assembles + installs the kernel at sim init
- Modify: `include/ter/kernels.hpp` — declare `install_default_kernels`
- Modify: `src/CMakeLists.txt`
- Create: `tests/test_kernel_matmul_b_small.cpp`
- Modify: `tests/CMakeLists.txt`

The kernel computes `Y[i] = sum over k of X[k] * W[k]` for fixed-K = 27 (one tvmac chunk). Args in R1..R5: addr_X, addr_W, addr_Y, K, n_outputs (M = n outputs, N inner = K).

For F4.2 we keep the kernel narrow: dot product of two vectors of length 27 (one tvmac), result written to mem[addr_Y]. Multi-tile loops are added in the next task.

- [ ] **Step 1: Write `src/kernels/tk_matmul_b_9t.tasm`**

```
        ; tk_matmul_b_9t — single dot product of length 27.
        ; Args: r1 = addr_X (sim memory base of X), r2 = addr_W, r3 = addr_Y.
        ; Loads V0=X, V1=W, computes A0 = sum lanes (X*W), stores result at mem[r3].
        tvload     v0, r1
        tvload     v1, r2
        tvmac      a0, v0, v1
        tvsum      r4, a0
        tstore     r4, r3
        thalt
```

- [ ] **Step 2: Write `src/kernels/registry.cpp`**

```cpp
#include <ter/kernels.hpp>
#include <ter/sim.hpp>
#include <ter/assembler.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ter {

static std::string read_text(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("registry: cannot open " + path);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

void install_default_kernels(Sim& sim, KernelTable& kt, const std::string& kernels_dir) {
    auto src = read_text(kernels_dir + "/tk_matmul_b_9t.tasm");
    auto blob = assemble(src);
    kt.install(sim, "tk_matmul_b_9t", blob);
}

}  // namespace ter
```

- [ ] **Step 3: Append to `include/ter/kernels.hpp`**

```cpp
// Loads + assembles all built-in kernels into the sim and registers them.
// kernels_dir: filesystem path containing the .tasm sources.
void install_default_kernels(Sim& sim, KernelTable& kt, const std::string& kernels_dir);
```

- [ ] **Step 4: Write `tests/test_kernel_matmul_b_small.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/numfmt.hpp>
#include <vector>

using namespace ter;

TEST_CASE("tk_matmul_b_9t computes a single 27-length dot product") {
    Sim s(1024);
    KernelTable kt;
    install_default_kernels(s, kt, "src/kernels");
    KernelId id = kt.find("tk_matmul_b_9t");
    REQUIRE(id.valid);

    // Place X and W as int32 (lane values) at sim addresses 200 and 300.
    constexpr int K = 27;
    std::vector<int> X(K), W(K);
    for (int i = 0; i < K; ++i) { X[i] = i - 13; W[i] = (i % 5) - 2; }
    for (int i = 0; i < K; ++i) {
        s.mem().store_word(200 + i, Word27::from_int(X[i]));
        s.mem().store_word(300 + i, Word27::from_int(W[i]));
    }
    int Y_addr = 500;

    int64_t expected = 0;
    for (int i = 0; i < K; ++i) expected += int64_t{X[i]} * int64_t{W[i]};

    std::vector<int64_t> args = {200, 300, Y_addr, 0, 0, 0, 0};
    s.call_kernel(kt, id, args);

    int64_t got = s.mem().load_word(static_cast<size_t>(Y_addr)).to_int();
    CHECK(got == expected);

    // Counter check: exactly one tvmac, one tvsum.
    CHECK(s.counters().get(Opcode::TVMAC) == 1);
    CHECK(s.counters().get(Opcode::TVSUM) == 1);
}
```

- [ ] **Step 5: Wire build**

In `src/CMakeLists.txt` add `kernels/registry.cpp`.

In `tests/CMakeLists.txt`:
- Append `ter_add_test(test_kernel_matmul_b_small)`.
- Override its working dir to source root so `src/kernels/...` resolves:
```cmake
set_tests_properties(test_kernel_matmul_b_small PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
```

- [ ] **Step 6: Build and run**

Expected: 26/26.

- [ ] **Step 7: Commit**

```
git add src/kernels include/ter/kernels.hpp src/CMakeLists.txt tests/test_kernel_matmul_b_small.cpp tests/CMakeLists.txt
git commit -m "feat(kernels): tk_matmul_b_9t single-tile dot product + KernelTable loader"
```

---

## Task F4.3 — Multi-tile matmul: host orchestrates K-tiling + scaling

**Files:**
- Modify: `tests/test_kernel_matmul_b_small.cpp` — add a multi-tile case
- (No kernel change — the host loops over tiles, invoking the kernel once per inner-K=27 chunk)

For F4.3 we keep the kernel single-tile (length-27 dot product). The host wraps it in two loops (M outputs × ceil(K/27) inner tiles), accumulating partial sums and applying the final `scale_X * scale_W` on the host side. This proves the K3 split: heavy math in ternary kernel, orchestration on host.

- [ ] **Step 1: Append to `tests/test_kernel_matmul_b_small.cpp`**

```cpp
TEST_CASE("multi-tile matmul: 1x54 @ 54x1 via two tiles") {
    Sim s(2048);
    KernelTable kt;
    install_default_kernels(s, kt, "src/kernels");
    KernelId id = kt.find("tk_matmul_b_9t");

    constexpr int K = 54;
    std::vector<int> X(K), W(K);
    for (int i = 0; i < K; ++i) { X[i] = (i * 7) % 19 - 9; W[i] = (i * 5) % 13 - 6; }

    int xa = 100, wa = 1000, ya = 2000;
    for (int i = 0; i < K; ++i) {
        s.mem().store_word(static_cast<size_t>(xa + i), Word27::from_int(X[i]));
        s.mem().store_word(static_cast<size_t>(wa + i), Word27::from_int(W[i]));
    }

    int64_t expected = 0;
    for (int i = 0; i < K; ++i) expected += int64_t{X[i]} * int64_t{W[i]};

    int64_t got = 0;
    for (int t = 0; t < 2; ++t) {
        int tile_x = xa + t * 27;
        int tile_w = wa + t * 27;
        std::vector<int64_t> args = {tile_x, tile_w, ya, 0, 0, 0, 0};
        s.call_kernel(kt, id, args);
        got += s.mem().load_word(static_cast<size_t>(ya)).to_int();
    }

    CHECK(got == expected);
    CHECK(s.counters().get(Opcode::TVMAC) == 2);
}
```

- [ ] **Step 2: Build and run**

Expected: 26/26 (same test count, more CHECKs).

- [ ] **Step 3: Commit**

```
git add tests/test_kernel_matmul_b_small.cpp
git commit -m "test(kernels): multi-tile matmul orchestration via host"
```

---

## Task F4.4 — End-to-end format-B matmul vs numpy with bounded relative error

**Files:**
- Create: `tools/matmul_b_reference.py` — generates random float matrices + numpy reference
- Create: `tests/test_kernel_matmul_b_32.cpp`
- Modify: `tests/CMakeLists.txt`

This is the F4.4 deliverable: take random float matrices, quantize them to format B, compute the matmul through the ternary kernel, dequantize the result via per-tensor scales, and confirm relative error vs the numpy float reference is below a threshold.

- [ ] **Step 1: Write `tools/matmul_b_reference.py`**

```python
#!/usr/bin/env python3
"""Generates random float matrices A (M,K) and B (K,N) plus C = A@B for the format-B matmul test."""
import argparse
import os
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="build/matmul_b_data")
    p.add_argument("--seed", type=int, default=0xBEE)
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--k", type=int, default=27)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    A = rng.standard_normal((a.m, a.k)).astype(np.float32)
    B = rng.standard_normal((a.k, a.n)).astype(np.float32)
    C = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
    A.tofile(os.path.join(a.out_dir, "A.bin"))
    B.tofile(os.path.join(a.out_dir, "B.bin"))
    C.tofile(os.path.join(a.out_dir, "C.bin"))
    with open(os.path.join(a.out_dir, "shape.txt"), "w") as f:
        f.write(f"{a.m} {a.n} {a.k}\n")
    print("wrote A,B,C,shape to", a.out_dir)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `tests/test_kernel_matmul_b_32.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/numfmt.hpp>
#include <fstream>
#include <vector>
#include <cmath>

using namespace ter;

static std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.is_open());
    f.seekg(0, std::ios::end);
    size_t n = static_cast<size_t>(f.tellg()) / 4;
    f.seekg(0);
    std::vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4));
    return v;
}

TEST_CASE("format-B matmul via kernel matches numpy within bounded rel_err") {
    int M, N, K;
    {
        std::ifstream f("matmul_b_data/shape.txt"); REQUIRE(f.is_open());
        f >> M >> N >> K;
    }
    auto A = read_f32("matmul_b_data/A.bin");
    auto B = read_f32("matmul_b_data/B.bin");
    auto C_ref = read_f32("matmul_b_data/C.bin");

    // Quantize A row-by-row using a single per-tensor scale (paper-grade simplification).
    TritTensor At = quantize(A.data(), {M, K}, 9);
    TritTensor Bt = quantize(B.data(), {K, N}, 9);

    Sim s(8 * 1024);
    KernelTable kt;
    install_default_kernels(s, kt, "src/kernels");
    KernelId id = kt.find("tk_matmul_b_9t");

    // Place At and Bt in sim memory.
    int x_base = 1000, w_base = 4000, y_addr = 7000;
    for (int idx = 0; idx < M * K; ++idx) s.mem().store_word(x_base + idx, At.payload[idx]);
    for (int idx = 0; idx < K * N; ++idx) s.mem().store_word(w_base + idx, Bt.payload[idx]);

    std::vector<float> C(M * N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int64_t int_acc = 0;
            for (int k0 = 0; k0 < K; k0 += 27) {
                int chunk = std::min(27, K - k0);
                if (chunk < 27) {
                    // Pad with zeros: copy slice + zeros into a scratch region.
                    int scratch_x = 9000;
                    int scratch_w = 9100;
                    for (int t = 0; t < 27; ++t) {
                        int xv = t < chunk ? At.payload[i * K + k0 + t].to_int() : 0;
                        int wv = t < chunk ? Bt.payload[(k0 + t) * N + j].to_int() : 0;
                        s.mem().store_word(scratch_x + t, Word27::from_int(xv));
                        s.mem().store_word(scratch_w + t, Word27::from_int(wv));
                    }
                    std::vector<int64_t> args = {scratch_x, scratch_w, y_addr, 0, 0, 0, 0};
                    s.call_kernel(kt, id, args);
                } else {
                    // Full tile: gather B column k0..k0+27 into a contiguous slot.
                    int scratch_w = 9100;
                    for (int t = 0; t < 27; ++t) {
                        int wv = Bt.payload[(k0 + t) * N + j].to_int();
                        s.mem().store_word(scratch_w + t, Word27::from_int(wv));
                    }
                    std::vector<int64_t> args = {x_base + i * K + k0, scratch_w, y_addr, 0, 0, 0, 0};
                    s.call_kernel(kt, id, args);
                }
                int_acc += s.mem().load_word(static_cast<size_t>(y_addr)).to_int();
            }
            C[i * N + j] = static_cast<float>(int_acc) * At.scale * Bt.scale;
        }
    }

    // Bounded relative error per element vs numpy reference.
    double max_rel = 0.0;
    for (int idx = 0; idx < M * N; ++idx) {
        double ref = static_cast<double>(C_ref[idx]);
        double got = static_cast<double>(C[idx]);
        double denom = std::max(1.0, std::fabs(ref));
        double rel = std::fabs(got - ref) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    // 9 trits per element with K=27 contributors: expect rel_err well under 1e-2.
    CHECK(max_rel < 1e-2);

    // Counter check: exactly M * N * ceil(K/27) tvmacs.
    int tiles = (K + 26) / 27;
    CHECK(s.counters().get(Opcode::TVMAC) == static_cast<uint64_t>(M * N * tiles));
}
```

- [ ] **Step 3: Wire CMake fixture in `tests/CMakeLists.txt` (append):**

```cmake
add_test(NAME gen_matmul_b_data
    COMMAND ${CMAKE_SOURCE_DIR}/.venv/bin/python
            ${CMAKE_SOURCE_DIR}/tools/matmul_b_reference.py
            --out-dir ${CMAKE_BINARY_DIR}/matmul_b_data
            --m 8 --n 8 --k 27
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(gen_matmul_b_data PROPERTIES FIXTURES_SETUP matmul_b_data)

ter_add_test(test_kernel_matmul_b_32)
set_tests_properties(test_kernel_matmul_b_32 PROPERTIES
    FIXTURES_REQUIRED matmul_b_data
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
```

(Note the test reads `src/kernels/...tasm` via a relative path. Since WORKING_DIRECTORY is `${CMAKE_BINARY_DIR}`, we must either absolutize the kernel path or symlink. Simpler: have the test use the absolute source path. Update step 2's `install_default_kernels` call to use `CMAKE_SOURCE_DIR/src/kernels` injected via a generated header, OR pass it as a compile-time define.)

Actually the simplest fix: in `tests/CMakeLists.txt`, add a definition for the test:
```cmake
target_compile_definitions(test_kernel_matmul_b_32 PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
```
and in the test, replace `"src/kernels"` with `TER_KERNELS_DIR`. Apply the same to `test_kernel_matmul_b_small` so both tests share the convention.

Update `test_kernel_matmul_b_small.cpp` Step 4 install call:
```cpp
install_default_kernels(s, kt, TER_KERNELS_DIR);
```
And in `tests/CMakeLists.txt`:
```cmake
target_compile_definitions(test_kernel_matmul_b_small PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
```

- [ ] **Step 4: Build and run**

```
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: 28/28 (24 prior + new fixture + new test, plus the test_kernel_matmul_b_small now uses the absolute path so we can drop the WORKING_DIRECTORY override on it).

- [ ] **Step 5: Commit**

```
git add tools/matmul_b_reference.py tests/test_kernel_matmul_b_32.cpp tests/test_kernel_matmul_b_small.cpp tests/CMakeLists.txt
git commit -m "test(kernels): F4.4 — format-B matmul via kernel matches numpy within bounded rel_err"
```

---

## Final Task — Update README and ISA docs

**Files:**
- Modify: `README.md` — mark F3 / F4 partial done
- Modify: `docs/isa.md` — note that no new opcodes were added (F3/F4 are quantizer + kernel orchestration)
- Create: `docs/number-formats.md` — document format B and the per-tensor scale convention

- [ ] **Step 1: Update `README.md` Status section**

Replace the Status block with:

```markdown
## Status
- [x] F0 — Trit, Tryte, Word27, Word54 primitives, packing, Memory.
- [x] F1 — Scalar ISA, assembler, simulator, sum(1..5) smoke test.
- [x] F2 — SIMD extension (tvadd, tvmac, tvsum, ...), 64x64 matmul gate.
- [x] F3 — Format B quantizer (bf16/float ↔ fixed-point trit + per-tensor scale).
- [x] F4 (partial) — Sim::call_kernel, KernelTable, tk_matmul_b_9t kernel; format-B matmul vs numpy gate.
- [ ] F4 (rest) — RMSNorm, RoPE, SwiGLU, softmax, attention kernels (next plan).
- [ ] F5 — ntransformer bridge.
- [ ] F6 — Llama 3.2 1B end-to-end.
```

- [ ] **Step 2: Create `docs/number-formats.md`**

```markdown
# ter — Number Formats

## Format B — fixed-point ternary + per-tensor scale (MVP)

Each tensor stores:
- `dtype = TritFP_B`
- `n_trits_per_elem` (default 9)
- `scale: float32` (per tensor)
- `shape`
- `payload`: one Word27 per element, lower `n_trits_per_elem` trits valid

### Conversion bf16 → trit

```
mti      = (3^n_trits_per_elem - 1) / 2     # max representable trit-int
scale    = max(|tensor|) / mti              # 0 if tensor is all-zero
trit_int = round(value / scale)             # clamped to [-mti, +mti]
trits    = balanced_ternary_digits(trit_int, n_trits_per_elem)
```

### Quality budget

With `n_trits_per_elem = 9`:
- Effective bits ≈ log2(3^9) = 14.27 — between int10 and int15.
- Quantization noise variance ≤ scale²/12 per element.
- Round-trip MSE on uniform `[-1, 1]` test data is bounded by `4·scale²/12` (see `test_numfmt_roundtrip.cpp`).

### Matmul under format B

For `Y = X · W^T`:
1. `acc_int = sum_k X.payload[i,k].to_int() * W.payload[k,j].to_int()` — pure integer ternary, computed by `tvmac` chunks.
2. `Y[i,j] = acc_int * X.scale * W.scale` — single float multiply per output, on the host.

Inside the simulator there are zero floating-point operations. The float scale lives on the host, applied once per output tile. This is the operational embodiment of the project's thesis: matmul reduces to ternary additions and a single per-tile float scale.

## Format A — `tfloat` (deferred)

Native ternary float (1-trit sign · 5-trit exponent · 9-trit mantissa). Documented for the optional F7 phase.
```

- [ ] **Step 3: Build and verify all tests still pass**

```
cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: same as before; only docs changed.

- [ ] **Step 4: Commit**

```
git add README.md docs/number-formats.md
git commit -m "docs(ter): F3 + F4-partial done — format B reference + status update"
```

---

## Self-Review

- **Spec coverage:** §7.1 (Format B) covered by F3.1-F3.4. §8 (K3 kernels) — `Sim::call_kernel` + KernelTable + first kernel `tk_matmul_b_9t` covered by F4.1-F4.4. §9 (ntransformer bridge), §10 (validation gates for full Llama 1B), §11.bis (K4) explicitly deferred.
- **Placeholder scan:** no TBDs. The kernel `tk_matmul_b_9t` is intentionally single-tile (27-length dot product); multi-tile orchestration lives on the host. Documented.
- **Type consistency:** `DType`, `TritTensor`, `KernelId`, `KernelTable`, `quantize`, `dequantize`, `install_default_kernels`, `Sim::call_kernel`, `tk_matmul_b_9t` referenced consistently across all tasks.
- **Known caveat:** `test_kernel_matmul_b_32` uses simple per-tensor scale and a small (8×8×27) shape. Larger shapes (e.g. real transformer dimensions like 2048×2048) wait until full F4 lands and we have RMSNorm-style channel-wise scale or empirical evidence that per-tensor scale suffices.

---

## Execution Handoff

After all tasks complete, the project has:
- Format B quantizer with documented bounds.
- A first ternary bytecode kernel running in the simulator, invoked from host.
- An end-to-end test that closes the loop: floats → quantize → ternary kernel → dequantize → matches numpy.

Next plan: implement the remaining F4 kernels (rmsnorm, rope, swiglu, softmax, attention) — each one is ~80-150 lines of `.tasm` plus a vs-reference test.
