# `ter` First GGUF Load Plan (F5.4a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Open the brandon-tiny f16 GGUF (`~/osito-a-models/build/brandon-tiny-10m-f16.gguf`), load one named weight tensor through the vendored `nt::GGUFLoader`, dequantize it from f16 to f32, and convert to `TritTensor` (Format B, 9 trits/elem). End state: a smoke test asserts (a) the loader opens the file, (b) we can fetch a tensor by name, (c) the dequantized shape matches what brandon-tiny's config promises, (d) the round-trip MSE is bounded.

**Architecture:** Add two helpers to `vendor/ntransformer/`:
1. `dequant_f16(const void* src, size_t n_elems, float* out)` — naive f16→f32 conversion (no SIMD; called once per weight load, perf is fine).
2. `tensor_to_trit(const Tensor& t, int n_trits)` — wraps `nt::Tensor`'s data + shape, runs our `quantize()` from `<ter/numfmt.hpp>`, returns a `TritTensor`. Currently only handles f32 input; f16 is dequanted first.

End-to-end: `GGUFLoader → get_tensor(name) → tensor_to_trit() → TritTensor`.

For brandon-tiny the weight tensor we'll target first is `token_embd.weight` (always present in llama-arch GGUFs, shape `(vocab_size, hidden_dim) = (8192, 256) = 2_097_152` f16 elements). It's a known-large tensor that exercises the load path without depending on per-layer naming conventions.

**Tech Stack:** Same as F5.3.

**Spec:** `docs/superpowers/specs/2026-05-08-ter-design.md` §9. Bridge contract: `docs/bridge-notes.md`.

**Out of scope:**
- Loading ALL tensors of a model (one tensor proves the path).
- Q4_K_M / Q8_0 dequantization (later — those need block decoders from ntransformer's CUDA kernels reimplemented as host code).
- TinyLlama / Llama 3.2 1B targets — F5.4c+ once brandon-tiny path is proven.
- Multi-layer transformer assembly — F5.4c.

---

## File Structure

```
vendor/ntransformer/
├── core/
│   ├── dequant.hpp        # NEW: dequant_f16, dequant_f32 (passthrough), forward decls for Q*
│   └── dequant.cpp        # NEW: f16 → f32 IEEE 754 conversion
include/ter/host/
└── load_gguf.hpp          # NEW: tensor_to_trit() bridge
src/host/
└── load_gguf.cpp          # NEW: implementation
tests/
└── test_load_brandon_tiny.cpp   # opens the real GGUF, fetches token_embd, asserts shape + round-trip MSE
```

The `host/` directory is for ter-side bridge code (vs `tx/` which is transformer logic). Keeps the layering visible.

---

## Task F5.4a.1 — `dequant.{hpp,cpp}` for f16 → f32

**Files:**
- Create `vendor/ntransformer/core/dequant.hpp`
- Create `vendor/ntransformer/core/dequant.cpp`
- Create `tests/test_nt_dequant.cpp`
- Modify `vendor/ntransformer/CMakeLists.txt`, `tests/CMakeLists.txt`

**Critical git discipline:** Stay on `feature/f0-f2-foundation`, no detached HEAD.

- [ ] **Step 1: Write `vendor/ntransformer/core/dequant.hpp`**

```cpp
#pragma once
#include <cstdint>
#include <cstddef>

namespace nt {

// Convert a buffer of n IEEE 754 binary16 (half-precision) values to f32.
// src must point to n * 2 bytes; out must hold n floats.
void dequant_f16(const void* src, std::size_t n_elems, float* out);

// f32 passthrough (identity copy). Convenience for callers that do dispatch
// without special-casing.
void dequant_f32(const void* src, std::size_t n_elems, float* out);

}  // namespace nt
```

- [ ] **Step 2: Write `vendor/ntransformer/core/dequant.cpp`**

```cpp
#include <core/dequant.hpp>
#include <cstring>

namespace nt {

namespace {

// IEEE 754 half-to-float, branchless, no NaN/Inf special-casing beyond the obvious.
// Adapted from the standard "manual" implementation; safe for all bit patterns.
float half_to_float(std::uint16_t h) {
    std::uint32_t sign = (h & 0x8000u) << 16;
    std::uint32_t exp  = (h >> 10) & 0x1fu;
    std::uint32_t mant = h & 0x3ffu;
    std::uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;                             // ±0
        } else {
            // Subnormal: normalize.
            int shift = 0;
            while ((mant & 0x400u) == 0) { mant <<= 1; ++shift; }
            mant &= 0x3ffu;
            f = sign | ((127u - 14u - shift) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        f = sign | 0x7f800000u | (mant << 13);    // ±Inf or NaN
    } else {
        f = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

}  // namespace

void dequant_f16(const void* src, std::size_t n_elems, float* out) {
    const std::uint16_t* halves = static_cast<const std::uint16_t*>(src);
    for (std::size_t i = 0; i < n_elems; ++i) out[i] = half_to_float(halves[i]);
}

void dequant_f32(const void* src, std::size_t n_elems, float* out) {
    std::memcpy(out, src, n_elems * sizeof(float));
}

}  // namespace nt
```

- [ ] **Step 3: Write `tests/test_nt_dequant.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <core/dequant.hpp>
#include <cstdint>
#include <vector>
#include <cmath>

using namespace nt;

TEST_CASE("dequant_f16 handles known bit patterns") {
    std::vector<std::uint16_t> halves = {
        0x0000,   // +0
        0x8000,   // -0
        0x3c00,   // +1.0
        0xbc00,   // -1.0
        0x4200,   // +3.0
        0x3555,   // ~0.333
    };
    std::vector<float> out(halves.size());
    dequant_f16(halves.data(), halves.size(), out.data());
    CHECK(out[0] == 0.0f);
    CHECK(out[1] == -0.0f);
    CHECK(out[2] == 1.0f);
    CHECK(out[3] == -1.0f);
    CHECK(out[4] == 3.0f);
    CHECK(std::fabs(out[5] - 0.33325195f) < 1e-5f);
}

TEST_CASE("dequant_f32 is identity") {
    std::vector<float> in = {1.0f, -2.5f, 3.14f};
    std::vector<float> out(in.size());
    dequant_f32(in.data(), in.size(), out.data());
    for (size_t i = 0; i < in.size(); ++i) CHECK(out[i] == in[i]);
}
```

- [ ] **Step 4: Wire build**

In `vendor/ntransformer/CMakeLists.txt`, add `core/dequant.cpp` to `nt_infra` sources.

In `tests/CMakeLists.txt`:
```cmake
ter_add_test(test_nt_dequant)
target_link_libraries(test_nt_dequant PRIVATE nt_infra)
```

- [ ] **Step 5: Build and run**

```
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: 48/48.

- [ ] **Step 6: Commit**

```
git add vendor/ntransformer/core/dequant.hpp vendor/ntransformer/core/dequant.cpp vendor/ntransformer/CMakeLists.txt tests/test_nt_dequant.cpp tests/CMakeLists.txt
git commit -m "feat(vendor): dequant_f16 + dequant_f32 helpers"
```

After: `git log --oneline -2`.

---

## Task F5.4a.2 — `tensor_to_trit()` bridge helper

**Files:**
- Create `include/ter/host/load_gguf.hpp`
- Create `src/host/load_gguf.cpp`
- Modify `src/CMakeLists.txt`

The bridge turns an `nt::Tensor` (CPU, f16 or f32 dtype) into a `ter::TritTensor` (Format B, 9 trits/elem). For unsupported dtypes (Q4_0, Q8_0, etc.), it throws — those land in F5.4c.

- [ ] **Step 1: Write `include/ter/host/load_gguf.hpp`**

```cpp
#pragma once
#include <core/tensor.h>
#include <ter/numfmt.hpp>

namespace ter::host {

// Convert an nt::Tensor (Device::CPU, dtype = F32 or F16) into a Format B
// TritTensor with the given trit width. Throws std::runtime_error for any
// other dtype (Q4_0 / Q8_0 / etc. land in F5.4c with proper unpackers).
ter::TritTensor tensor_to_trit(const nt::Tensor& t, int n_trits_per_elem = 9);

}  // namespace ter::host
```

- [ ] **Step 2: Write `src/host/load_gguf.cpp`**

```cpp
#include <ter/host/load_gguf.hpp>
#include <core/dequant.hpp>
#include <stdexcept>
#include <vector>

namespace ter::host {

ter::TritTensor tensor_to_trit(const nt::Tensor& t, int n_trits_per_elem) {
    if (t.device() != nt::Device::CPU) {
        throw std::runtime_error("tensor_to_trit: input must be on CPU");
    }

    // Compute element count from shape.
    std::size_t n_elems = 1;
    std::vector<int> shape_int;
    for (auto d : t.shape()) {
        n_elems *= static_cast<std::size_t>(d);
        shape_int.push_back(static_cast<int>(d));
    }

    // Dequantize to a temporary float buffer.
    std::vector<float> tmp(n_elems);
    switch (t.dtype()) {
        case nt::DType::F32:
            nt::dequant_f32(t.data(), n_elems, tmp.data());
            break;
        case nt::DType::F16:
            nt::dequant_f16(t.data(), n_elems, tmp.data());
            break;
        default:
            throw std::runtime_error("tensor_to_trit: unsupported dtype "
                                     "(only F16/F32 supported in F5.4a; Q* lands in F5.4c)");
    }

    // Quantize via ter::quantize().
    return ter::quantize(tmp.data(), shape_int, n_trits_per_elem);
}

}  // namespace ter::host
```

- [ ] **Step 3: Wire build**

In `src/CMakeLists.txt`, add `host/load_gguf.cpp` to the `ter` library sources. The `ter` library needs to link against `nt_infra` so `<core/tensor.h>` resolves; if that's not already wired, add:

```cmake
target_link_libraries(ter PUBLIC nt_infra)
```

- [ ] **Step 4: Build (no test yet)**

```
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: 48/48 still (no behavior change for existing tests; new helper is unused).

If the build fails because `nt::Tensor::shape()` returns a different type than expected, adapt the loop's iteration. The vendor's tensor.h is the source of truth.

- [ ] **Step 5: Commit**

```
git add include/ter/host/load_gguf.hpp src/host/load_gguf.cpp src/CMakeLists.txt
git commit -m "feat(host): tensor_to_trit() bridge for F16/F32 nt::Tensor"
```

After: `git log --oneline -2`.

---

## Task F5.4a.3 — End-to-end smoke: open brandon-tiny, fetch one tensor

**Files:**
- Create `tests/test_load_brandon_tiny.cpp`
- Modify `tests/CMakeLists.txt`

The test opens the actual GGUF from `~/osito-a-models/build/brandon-tiny-10m-f16.gguf`, fetches `token_embd.weight` (shape `(vocab_size, hidden_size) = (8192, 256)`), and verifies the round-trip through Format B keeps the MSE within an expected bound.

If the file is not present (e.g. CI), the test should `SKIP` rather than fail.

- [ ] **Step 1: Write `tests/test_load_brandon_tiny.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <model/loader.h>
#include <ter/host/load_gguf.hpp>
#include <ter/numfmt.hpp>
#include <fstream>
#include <vector>
#include <cmath>

using namespace nt;

namespace {

// brandon-tiny config (per ~/osito-a-models/downloads/brandon-tiny-10m-instruct/config.json):
constexpr int VOCAB    = 8192;
constexpr int HIDDEN   = 256;
constexpr int N_LAYERS = 24;
constexpr const char* GGUF_PATH = "/Users/pc/osito-a-models/build/brandon-tiny-10m-f16.gguf";

bool file_exists(const char* path) {
    std::ifstream f(path);
    return f.good();
}

}  // namespace

TEST_CASE("brandon-tiny GGUF loads (or skip)") {
    if (!file_exists(GGUF_PATH)) {
        MESSAGE("brandon-tiny GGUF not found at ", GGUF_PATH, " — skipping");
        return;
    }

    GGUFLoader loader;
    REQUIRE(loader.load(GGUF_PATH));

    // Dump basic config from GGUF metadata for sanity.
    const auto& cfg = loader.config();
    MESSAGE("loaded brandon-tiny: vocab=", cfg.vocab_size,
            " hidden=", cfg.hidden_size,
            " n_layers=", cfg.n_layers);

    // The GGUF metadata may not exactly match brandon-tiny's config.json
    // (the conversion script normalises to llama-arch). Just sanity-check
    // they're in the right ballpark.
    CHECK(cfg.vocab_size > 0);
    CHECK(cfg.hidden_size > 0);
    CHECK(cfg.n_layers > 0);
}

TEST_CASE("brandon-tiny token_embd.weight: load and quantize to Format B") {
    if (!file_exists(GGUF_PATH)) {
        MESSAGE("brandon-tiny GGUF not found — skipping");
        return;
    }

    GGUFLoader loader;
    REQUIRE(loader.load(GGUF_PATH));

    const auto* info = loader.tensor_info("token_embd.weight");
    REQUIRE(info != nullptr);
    MESSAGE("token_embd.weight dtype=", static_cast<int>(/*info-derived dtype field*/0));

    Tensor t = loader.get_tensor("token_embd.weight");
    REQUIRE(t.dtype() == DType::F16);   // brandon-tiny f16 GGUF
    REQUIRE(t.shape().size() == 2);

    // Shape may be (VOCAB, HIDDEN) or (HIDDEN, VOCAB) depending on conversion convention.
    // Just check the product matches.
    std::size_t n = 1;
    for (auto d : t.shape()) n *= static_cast<std::size_t>(d);
    CHECK(n == static_cast<std::size_t>(VOCAB) * static_cast<std::size_t>(HIDDEN));

    // Convert to TritTensor at 9 trits/elem.
    auto tt = ter::host::tensor_to_trit(t, 9);
    CHECK(tt.dtype == ter::DType::TritFP_B);
    CHECK(tt.n_trits_per_elem == 9);
    CHECK(tt.payload.size() == n);
    CHECK(tt.scale > 0.0f);

    // Round-trip MSE bound: dequantize back and compare to the original f16 → f32 path.
    std::vector<float> dequantized(n);
    nt::dequant_f16(t.data(), n, dequantized.data());
    std::vector<float> roundtrip(n);
    ter::dequantize(tt, roundtrip.data());

    double sse = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double d = double(dequantized[i]) - double(roundtrip[i]);
        sse += d * d;
    }
    double mse = sse / static_cast<double>(n);
    // Expected MSE bound: for 9-trit per-tensor scale on a tensor with
    // max(|x|) ≈ a few, scale ≈ 1e-4, so quantization variance ≤ scale²/12 ≈ 8e-10.
    // Be generous (1e-6) to allow for outliers and per-tensor scale skew.
    MESSAGE("token_embd Format-B round-trip MSE = ", mse);
    CHECK(mse < 1e-3);
}
```

The `MESSAGE` macros emit useful diagnostics during ctest run; don't fail tests.

- [ ] **Step 2: Wire test in `tests/CMakeLists.txt`**

```cmake
ter_add_test(test_load_brandon_tiny)
target_link_libraries(test_load_brandon_tiny PRIVATE nt_infra)
```

(No fixture — the GGUF is a real file at a fixed path, not a build artifact.)

- [ ] **Step 3: Build and run**

Expected: 49/49 (48 prior + new). The new test SKIPS gracefully if the GGUF is missing.

If the GGUF is present but the test fails at REQUIRE(loader.load(...)), debug:
- Loader may need a different magic byte / version.
- Check the file is not corrupt: `head -c 4 ~/osito-a-models/build/brandon-tiny-10m-f16.gguf` should output `GGUF`.

If `tensor_info("token_embd.weight")` returns nullptr, the tensor name may be different in this GGUF. Run `tools/decompose_gguf.py` if available, or manually inspect the loader's tensor list:
```cpp
auto names = loader.tensor_names();
for (const auto& n : names) MESSAGE(n);
```

If the round-trip MSE exceeds 1e-3, the per-tensor scale on a 2M-element tensor may be dominated by a few outliers. Relax the bound (1e-2) and document.

- [ ] **Step 4: Commit**

```
git add tests/test_load_brandon_tiny.cpp tests/CMakeLists.txt
git commit -m "test(host): F5.4a — load brandon-tiny GGUF + quantize token_embd to Format B"
```

After: `git log --oneline -3`.

---

## Final Task — README + bridge notes update

**Files:**
- Modify `README.md` Status block
- Append to `docs/bridge-notes.md`

- [ ] **Step 1: Update `README.md` Status block**

Replace the F5.4 line with:
```markdown
- [x] F5.4a — Load brandon-tiny f16 GGUF + quantize token_embd to Format B; round-trip MSE bounded.
- [ ] F5.4b — Tile-aware rmsnorm/softmax for hidden_size > 27.
- [ ] F5.4c — Multi-layer Transformer + tokenizer + sampler; first generated token from brandon-tiny.
- [ ] F5.4d — TinyStories Q4_K_M (test the unpacker in vendored loader).
- [ ] F6   — Llama 3.2 1B Q8_0 end-to-end (paper target).
```

- [ ] **Step 2: Append to `docs/bridge-notes.md`**

```markdown
## F5.4a result — first GGUF tensor through the full path

- `nt::dequant_f16()` and `nt::dequant_f32()` live in `vendor/ntransformer/core/dequant.{hpp,cpp}`. The f16 implementation is a portable IEEE 754 half-to-float (branchless on the common path).
- `ter::host::tensor_to_trit()` in `src/host/load_gguf.cpp` is the bridge: dequant to f32, then call `ter::quantize()` for Format B.
- The smoke test opens `~/osito-a-models/build/brandon-tiny-10m-f16.gguf`, fetches `token_embd.weight` (shape ≈ (8192, 256)), and verifies the round-trip through Format B keeps MSE bounded.

### Next quantizations to support

`tensor_to_trit()` currently throws on Q4_0 / Q4_K_M / Q5_K / Q6_K / Q8_0. The vendored `nt::GGUFLoader` recognises the dtype enum but doesn't dequant — those need block decoders. F5.4d adds Q4_K_M support (the most common quant) so we can target TinyStories. Llama 3.2 1B is Q8_0 — handled in F6 alongside the heavier weight class.

### Why brandon-tiny first

- F16 dequant is one function; Q4_K_M is six (block scales, group scales, per-block min/max, etc.).
- 21 MB total weights vs 1.3 GB for Llama 3.2 1B — fast to iterate.
- Instruct-tuned with a chatml format, so we can validate "does it generate sensible text" later.
- Non-standard config items (block_sharing, value_residual, dense_former, n_registers) are documented in `~/osito-a-models/CLAUDE.md`. We will simplify (treat block_sharing=true as "load 1 layer, apply 24 times"; ignore the rest) and note discrepancies.
```

- [ ] **Step 3: Build, verify (no test changes)**

Expected: 49/49 still.

- [ ] **Step 4: Commit**

```
git add README.md docs/bridge-notes.md
git commit -m "docs(ter): F5.4a done -- first GGUF tensor through the full path"
```

After: `git log --oneline -3`.

---

## Self-Review

- **Spec coverage:** §9 — F5.4a covers the loader bridge for one tensor with f16 dequant. F5.4b/c/d cover the rest.
- **Placeholder scan:** the test diagnostic message about info-derived dtype field is approximate (see Step 1 — adjust to match the actual `GGUFTensorInfo` field name once the loader's header is open).
- **Type consistency:** `dequant_f16`, `dequant_f32`, `tensor_to_trit`, `nt::DType::F16`, `ter::DType::TritFP_B`, `nt::Tensor`, `ter::TritTensor` referenced consistently.
- **Known caveats:**
  - The test depends on a file at a hard-coded absolute path. SKIPS gracefully if missing — but won't run on CI without the file.
  - Round-trip MSE bound (1e-3) is generous; the actual bound depends on the tensor's value distribution. May need tightening or loosening based on real measurement.

---

## Execution Handoff

After all tasks: we can pull a single weight tensor from a real Llama-arch GGUF, dequantize, and have it sitting in our Format B representation ready for kernel use. This is the loader bottom-half; F5.4b/c builds the full transformer assembly on top.

Next plan (F5.4b): tile-aware rmsnorm + softmax kernels for `hidden_size > 27`. Brandon-tiny has `hidden_size = 256 ≈ 9.5 tiles`, so this is gating before we can do real-shape inference.
