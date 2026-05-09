# `ter` Vendor ntransformer Infrastructure Plan (F5 part 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Lift the CUDA-independent infrastructure of `~/ntransformer` into `vendor/ntransformer/` inside `~/ter`, modify it minimally so it compiles without CUDA, and verify it works in isolation. End state: GGUF loader, Tensor abstraction, ModelConfig, Tokenizer, Sampler all live in `vendor/ntransformer/` and pass smoke tests. Sets the stage for F5.2 (our own transformer logic that calls ter's kernels).

**Architecture:** Lift-and-strip approach — copy source files from `/Users/pc/ntransformer/src/` into `/Users/pc/ter/vendor/ntransformer/`, replace any CUDA-coupled bits with CPU equivalents (no async streams, no `cudaMemcpy`, no F16 GPU buffers). Add `DType::TERNARY` to the dtype enum. Build as a static library linked into `ter`'s main binary. The `cuda::launch_*()` functions are NOT lifted — they're our problem to replace later.

**Tech Stack:** C++17, CMake, doctest, mmap (POSIX). No CUDA, no PyTorch.

**Spec:** §9 of `docs/superpowers/specs/2026-05-08-ter-design.md`. Bridge analysis: see prior `general-purpose` agent's exploration report.

**Out of scope for this plan:**
- Transformer/attention/ffn/norm logic — written fresh in F5.2 to call our kernels directly.
- The TernarySim dispatcher itself — F5.2.
- KV cache, sampling-loop, generation — F5.3.
- TinyLlama / Llama 3.2 1B end-to-end — F5.3 / F5.4.

---

## Why lift infrastructure but not transformer logic

The exploration report showed:
- Tensor / Config / Loader / Tokenizer / Sampler are CUDA-free or trivially de-CUDA-able.
- Transformer / Attention / FFN / Norm hardcode `cuda::launch_*()` everywhere with `void* stream`. Patching them means runtime branches at every call site.

Cleaner: lift the infrastructure (it's correct and battle-tested), write our own transformer code from scratch that calls our 5 kernels directly. That gives us a thin "Llama on ternary" front-end without inheriting the CUDA orchestration.

---

## File Structure (created during this plan)

```
ter/
├── vendor/ntransformer/
│   ├── CMakeLists.txt                       # builds nt_infra static lib
│   ├── core/
│   │   ├── types.{h,cpp}                    # DType enum + DType::TERNARY added
│   │   ├── tensor.{h,cpp}                   # CUDA-stripped
│   │   ├── allocator.{h,cpp}                # CPU-only malloc-backed
│   │   └── device.h                         # Device::CPU only (drop CUDA)
│   ├── model/
│   │   ├── config.{h,cpp}                   # drop-in
│   │   └── loader.{h,cpp}                   # mmap GGUF, CUDA-free
│   └── inference/
│       ├── tokenizer.{h,cpp}                # drop-in
│       └── sampler.{h,cpp}                  # drop-in
└── tests/
    ├── test_nt_tensor.cpp                   # round-trip Tensor create / access
    ├── test_nt_config.cpp                   # default fields, from_gguf_metadata stub
    ├── test_nt_loader.cpp                   # parse a tiny synthetic GGUF
    ├── test_nt_tokenizer.cpp                # encode/decode with a tiny vocab
    └── test_nt_sampler.cpp                  # deterministic sampling with a fixed seed
```

The `Engine` and `main.cpp` are NOT lifted — they're entangled with the transformer (which we're writing fresh) and the streaming pipeline (which we don't need).

---

## Task F5.1.1 — Lift `core/types.{h,cpp}` with `DType::TERNARY`

**Files:**
- Create `vendor/ntransformer/core/types.h`
- Create `vendor/ntransformer/core/types.cpp`
- Create `tests/test_nt_types.cpp`
- Create `vendor/ntransformer/CMakeLists.txt`
- Modify `CMakeLists.txt` (top-level), `tests/CMakeLists.txt`

**Critical git discipline:** Stay on `feature/f0-f2-foundation`, no detached HEAD.

- [ ] **Step 1: Read `/Users/pc/ntransformer/src/core/types.h` and `types.cpp`**

Read both files in full to understand the DType enum, dtype_size, dtype_block_size, ggml_to_dtype mappings.

- [ ] **Step 2: Create `vendor/ntransformer/core/types.h`**

Copy the original file's contents verbatim. Then add `TERNARY = 9` to the `DType` enum (the next free integer after the existing entries).

In `dtype_size()`, add `case DType::TERNARY: return 1;` (one byte per element — 9 trits packed loosely into a Word27 stored as uint64; we'll refine this when the dispatch lands in F5.2).

In `dtype_block_size()`, add `case DType::TERNARY: return 1;` (no blocking).

In `ggml_to_dtype()`, leave the function unchanged — `TERNARY` isn't a GGML type.

- [ ] **Step 3: Create `vendor/ntransformer/core/types.cpp`**

Copy verbatim. No changes needed.

- [ ] **Step 4: Create `vendor/ntransformer/CMakeLists.txt`**

```cmake
add_library(nt_infra STATIC
    core/types.cpp
)
target_include_directories(nt_infra PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(nt_infra PRIVATE -Wall -Wextra -Wpedantic -Werror)
```

- [ ] **Step 5: Wire into top-level `CMakeLists.txt`**

After `add_subdirectory(src)`, add:
```cmake
add_subdirectory(vendor/ntransformer)
```

- [ ] **Step 6: Create `tests/test_nt_types.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <core/types.h>

using namespace nt;

TEST_CASE("DType::TERNARY is present") {
    CHECK(static_cast<int>(DType::TERNARY) == 9);
}

TEST_CASE("dtype_size and dtype_block_size for TERNARY are non-zero") {
    CHECK(dtype_size(DType::TERNARY) > 0);
    CHECK(dtype_block_size(DType::TERNARY) > 0);
}

TEST_CASE("standard DType entries unchanged") {
    CHECK(dtype_size(DType::F32) == 4);
    CHECK(dtype_size(DType::F16) == 2);
    CHECK(dtype_block_size(DType::F32) == 1);
}
```

- [ ] **Step 7: Wire test in `tests/CMakeLists.txt`**

```cmake
ter_add_test(test_nt_types)
target_link_libraries(test_nt_types PRIVATE nt_infra)
```

- [ ] **Step 8: Build and run**

```
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: 39/39 (38 prior + new).

If types.cpp had CUDA-dependent code (unlikely; check), strip it.

- [ ] **Step 9: Commit**

```
git add vendor/ntransformer/core/types.h vendor/ntransformer/core/types.cpp vendor/ntransformer/CMakeLists.txt tests/test_nt_types.cpp tests/CMakeLists.txt CMakeLists.txt
git commit -m "feat(vendor): lift ntransformer core/types with DType::TERNARY"
```

---

## Task F5.1.2 — Lift `core/device.h` (CPU-only)

**Files:**
- Create `vendor/ntransformer/core/device.h`
- Modify `vendor/ntransformer/CMakeLists.txt`

- [ ] **Step 1: Read `/Users/pc/ntransformer/src/core/device.h`**

- [ ] **Step 2: Create `vendor/ntransformer/core/device.h`**

Copy the file verbatim, then strip:
- Anything that references `CUDA`, `cudaError`, `cudaStream`, etc.
- The `Device::CUDA` enum entry (we keep `Device::CPU` only)
- Any methods that depend on CUDA runtime

The result should be a tiny header that defines `enum class Device { CPU }` and any CPU-only helpers.

- [ ] **Step 3: Confirm `core/types.h` doesn't reference `Device::CUDA`**

If it does (e.g., in defaults), update those defaults to `Device::CPU`.

- [ ] **Step 4: Build and verify (no test changes)**

Expected: 39/39 still.

- [ ] **Step 5: Commit**

```
git add vendor/ntransformer/core/device.h
git commit -m "feat(vendor): lift core/device.h with CPU-only Device enum"
```

---

## Task F5.1.3 — Lift `core/tensor.{h,cpp}` (CUDA-stripped)

**Files:**
- Create `vendor/ntransformer/core/tensor.h`
- Create `vendor/ntransformer/core/tensor.cpp`
- Create `tests/test_nt_tensor.cpp`
- Modify `vendor/ntransformer/CMakeLists.txt`, `tests/CMakeLists.txt`

This is the largest single lift in this plan. The Tensor in ntransformer has CPU and CUDA paths — strip the CUDA paths.

- [ ] **Step 1: Read `/Users/pc/ntransformer/src/core/tensor.h` and `tensor.cpp` in full**

Note every CUDA dependency: `cudaMalloc`, `cudaMemcpy`, `cudaFree`, `cudaStream_t`, etc.

- [ ] **Step 2: Create `vendor/ntransformer/core/tensor.h`**

Copy verbatim with these surgical changes:
- Drop any `#include <cuda_runtime.h>` or similar.
- Drop any methods that take `cudaStream_t` or `void* stream` (or change them to no-op/CPU-only versions).
- Keep `to(Device target)` only for `target == Device::CPU` (a no-op).
- Keep `from_ptr`, `empty`, `zeros`, `view`, `slice`, `data`, `data_as`, `dtype`, `device`, `copy_from`.
- Allocator: switch to plain `malloc`/`free` instead of `cudaMalloc`/`cudaFree`.

- [ ] **Step 3: Create `vendor/ntransformer/core/tensor.cpp`**

Same surgical strip:
- Replace any `nt_cuda_malloc` / `nt_cuda_free` calls with `std::malloc` / `std::free`.
- Replace any `nt_cuda_memcpy_*` calls with `std::memcpy`.
- Drop any device-routing logic.

- [ ] **Step 4: Create `vendor/ntransformer/core/allocator.{h,cpp}`** (if Tensor depends on it)

If `tensor.cpp` uses an `Allocator` class, lift it too with a minimal CPU implementation. Otherwise skip.

- [ ] **Step 5: Create `tests/test_nt_tensor.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <core/tensor.h>
#include <vector>

using namespace nt;

TEST_CASE("Tensor::empty allocates with the requested shape and dtype") {
    auto t = Tensor::empty({4, 8}, DType::F32, Device::CPU);
    CHECK(t.dtype() == DType::F32);
    CHECK(t.device() == Device::CPU);
    CHECK(t.shape().size() == 2);
    CHECK(t.shape()[0] == 4);
    CHECK(t.shape()[1] == 8);
}

TEST_CASE("Tensor::zeros initialises bytes to zero") {
    auto t = Tensor::zeros({16}, DType::F32, Device::CPU);
    auto* p = t.data_as<float>();
    for (int i = 0; i < 16; ++i) CHECK(p[i] == 0.0f);
}

TEST_CASE("Tensor::from_ptr wraps without owning") {
    std::vector<float> buf = {1.0f, 2.0f, 3.0f, 4.0f};
    auto t = Tensor::from_ptr(buf.data(), {4}, DType::F32, Device::CPU);
    CHECK(t.data_as<float>()[2] == 3.0f);
}
```

- [ ] **Step 6: Wire build, register**

In `vendor/ntransformer/CMakeLists.txt`, add `core/tensor.cpp` (and `core/allocator.cpp` if created).

In `tests/CMakeLists.txt`, add:
```cmake
ter_add_test(test_nt_tensor)
target_link_libraries(test_nt_tensor PRIVATE nt_infra)
```

- [ ] **Step 7: Build and run**

If compile errors come from CUDA include or symbol references that you missed, strip them.

Expected: 40/40.

- [ ] **Step 8: Commit**

```
git add vendor/ntransformer/core tests/test_nt_tensor.cpp tests/CMakeLists.txt vendor/ntransformer/CMakeLists.txt
git commit -m "feat(vendor): lift core/tensor (CUDA-stripped, CPU-only)"
```

---

## Task F5.1.4 — Lift `model/config.{h,cpp}`

**Files:**
- Create `vendor/ntransformer/model/config.h`
- Create `vendor/ntransformer/model/config.cpp`
- Create `tests/test_nt_config.cpp`
- Modify `vendor/ntransformer/CMakeLists.txt`, `tests/CMakeLists.txt`

`ModelConfig` is a plain struct with safe defaults. Almost certainly drop-in.

- [ ] **Step 1: Read `/Users/pc/ntransformer/src/model/config.h` and `config.cpp`**

- [ ] **Step 2: Create `vendor/ntransformer/model/config.h`** — verbatim copy

- [ ] **Step 3: Create `vendor/ntransformer/model/config.cpp`** — verbatim copy

If `config.cpp` includes a CUDA header, drop the include. The body is pure C++ logic.

- [ ] **Step 4: Create `tests/test_nt_config.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <model/config.h>

using namespace nt;

TEST_CASE("ModelConfig defaults are sensible") {
    ModelConfig c;
    CHECK(c.vocab_size == 32000);
    CHECK(c.hidden_size == 4096);
    CHECK(c.n_heads == 32);
    CHECK(c.is_gqa() == false);
    CHECK(c.group_size() == 1);
}

TEST_CASE("GQA detection works") {
    ModelConfig c;
    c.n_heads = 32;
    c.n_kv_heads = 8;
    CHECK(c.is_gqa());
    CHECK(c.group_size() == 4);
}
```

- [ ] **Step 5: Wire and run**

In CMakeLists: add `model/config.cpp` to nt_infra.
Register `ter_add_test(test_nt_config)` linked to `nt_infra`.

Expected: 41/41.

- [ ] **Step 6: Commit**

```
git add vendor/ntransformer/model tests/test_nt_config.cpp tests/CMakeLists.txt vendor/ntransformer/CMakeLists.txt
git commit -m "feat(vendor): lift model/config (drop-in)"
```

---

## Task F5.1.5 — Lift `model/loader.{h,cpp}` (mmap GGUF, CUDA-free)

**Files:**
- Create `vendor/ntransformer/model/loader.h`
- Create `vendor/ntransformer/model/loader.cpp`
- Create `tests/test_nt_loader.cpp`
- Modify `vendor/ntransformer/CMakeLists.txt`, `tests/CMakeLists.txt`

The loader uses `mmap()` and is dtype-agnostic. Should lift cleanly.

- [ ] **Step 1: Read `/Users/pc/ntransformer/src/model/loader.h` and `loader.cpp`**

- [ ] **Step 2: Create `vendor/ntransformer/model/loader.h`** — verbatim, drop any CUDA include

- [ ] **Step 3: Create `vendor/ntransformer/model/loader.cpp`** — verbatim, replace:
- Any CUDA-related allocator usage with `malloc`/`free` (Tensor's `from_ptr` should not need to touch CUDA).
- Any device routing for `get_tensor()` — return CPU tensors only.

- [ ] **Step 4: Create a tiny synthetic GGUF for testing**

`tools/make_tiny_gguf.py`:

```python
#!/usr/bin/env python3
"""Generates a minimal valid GGUF file with one F32 tensor for testing."""
import argparse, os, struct

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

def write_string(f, s):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)

def write_kv_string(f, key, value):
    write_string(f, key)
    f.write(struct.pack("<I", 8))   # GGUF_TYPE_STRING
    write_string(f, value)

def write_kv_uint32(f, key, value):
    write_string(f, key)
    f.write(struct.pack("<I", 4))   # GGUF_TYPE_UINT32
    f.write(struct.pack("<I", value))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="build/tiny.gguf")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", 1))   # tensor_count
        f.write(struct.pack("<Q", 1))   # metadata_kv_count
        write_kv_string(f, "general.architecture", "llama")
        # one F32 tensor of shape (4,)
        write_string(f, "test.tensor")
        f.write(struct.pack("<I", 1))   # n_dims
        f.write(struct.pack("<Q", 4))   # dim 0
        f.write(struct.pack("<I", 0))   # GGML_TYPE_F32
        f.write(struct.pack("<Q", 0))   # offset (after alignment)
        # data section (simplest: 4 floats)
        f.write(struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))
    print(f"wrote tiny GGUF to {a.out}")

if __name__ == "__main__":
    main()
```

This is a minimal valid GGUF. If the loader rejects it because of stricter parsing, simplify the test goal — just check that `load(invalid_path)` returns false, and skip the synthetic file.

- [ ] **Step 5: Create `tests/test_nt_loader.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <model/loader.h>

using namespace nt;

TEST_CASE("Loader rejects non-existent file") {
    GGUFLoader l;
    CHECK_FALSE(l.load("/nonexistent/path.gguf"));
}

TEST_CASE("Loader rejects path with bad magic") {
    GGUFLoader l;
    CHECK_FALSE(l.load("/etc/hosts"));
}
```

(For a real GGUF round-trip, see F5.1.X catch-all at the end. Keep this task minimal.)

- [ ] **Step 6: Wire and run**

Add `model/loader.cpp` to nt_infra. Register the test.

Expected: 42/42.

- [ ] **Step 7: Commit**

```
git add vendor/ntransformer/model/loader.h vendor/ntransformer/model/loader.cpp tests/test_nt_loader.cpp tests/CMakeLists.txt vendor/ntransformer/CMakeLists.txt
git commit -m "feat(vendor): lift model/loader (mmap GGUF, CUDA-free)"
```

---

## Task F5.1.6 — Lift `inference/tokenizer.{h,cpp}`

**Files:**
- Create `vendor/ntransformer/inference/tokenizer.h`
- Create `vendor/ntransformer/inference/tokenizer.cpp`
- Create `tests/test_nt_tokenizer.cpp`
- Modify `vendor/ntransformer/CMakeLists.txt`, `tests/CMakeLists.txt`

Drop-in per the report (no CUDA, no threads).

- [ ] **Step 1: Read `/Users/pc/ntransformer/src/inference/tokenizer.h` and `tokenizer.cpp`**

- [ ] **Step 2: Create `vendor/ntransformer/inference/tokenizer.h`** — verbatim

- [ ] **Step 3: Create `vendor/ntransformer/inference/tokenizer.cpp`** — verbatim

- [ ] **Step 4: Create `tests/test_nt_tokenizer.cpp`**

The tokenizer needs a `GGUFVocab` to init. For the smoke test, build a tiny in-memory vocab manually.

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <inference/tokenizer.h>
#include <model/loader.h>   // for GGUFVocab definition

using namespace nt;

TEST_CASE("Tokenizer initialises with empty vocab without crashing") {
    GGUFVocab vocab;
    Tokenizer t;
    t.init(vocab, /*bos*/1, /*eos*/2);
    CHECK(t.bos_id() == 1);
    CHECK(t.eos_id() == 2);
}
```

If `GGUFVocab` is in a separate header, include that. If `init()` crashes on empty vocab, change the test to construct a tiny dummy vocab with one token.

- [ ] **Step 5: Wire and run**

Add `inference/tokenizer.cpp` to nt_infra. Register the test.

Expected: 43/43.

- [ ] **Step 6: Commit**

```
git add vendor/ntransformer/inference/tokenizer.h vendor/ntransformer/inference/tokenizer.cpp tests/test_nt_tokenizer.cpp tests/CMakeLists.txt vendor/ntransformer/CMakeLists.txt
git commit -m "feat(vendor): lift inference/tokenizer (drop-in)"
```

---

## Task F5.1.7 — Lift `inference/sampler.{h,cpp}`

**Files:**
- Create `vendor/ntransformer/inference/sampler.h`
- Create `vendor/ntransformer/inference/sampler.cpp`
- Create `tests/test_nt_sampler.cpp`
- Modify `vendor/ntransformer/CMakeLists.txt`, `tests/CMakeLists.txt`

Drop-in.

- [ ] **Step 1: Read `/Users/pc/ntransformer/src/inference/sampler.h` and `sampler.cpp`**

- [ ] **Step 2: Create `vendor/ntransformer/inference/sampler.{h,cpp}`** — verbatim

- [ ] **Step 3: Create `tests/test_nt_sampler.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <inference/sampler.h>
#include <vector>

using namespace nt;

TEST_CASE("Sampler with temperature 0 returns argmax") {
    Sampler s;
    SamplingConfig cfg;
    cfg.temperature = 0.0f;
    cfg.seed = 42;
    s.init(cfg);
    std::vector<float> logits = {0.1f, 0.5f, 0.9f, 0.3f};
    int tok = s.sample(logits.data(), static_cast<int>(logits.size()));
    CHECK(tok == 2);
}
```

If `Sampler::init` / `Sampler::sample` have different signatures than guessed, adjust the test to match the real API (read the header to confirm).

- [ ] **Step 4: Wire and run**

Add `inference/sampler.cpp` to nt_infra. Register the test.

Expected: 44/44.

- [ ] **Step 5: Commit**

```
git add vendor/ntransformer/inference/sampler.h vendor/ntransformer/inference/sampler.cpp tests/test_nt_sampler.cpp tests/CMakeLists.txt vendor/ntransformer/CMakeLists.txt
git commit -m "feat(vendor): lift inference/sampler (drop-in)"
```

---

## Final Task — Update README + first F5 patterns doc

**Files:**
- Modify `README.md` Status block
- Create `docs/bridge-notes.md` (new file documenting the bridge approach)

- [ ] **Step 1: Update `README.md` Status block**

Replace the F5 line with:
```markdown
- [x] F5.1 — vendor/ntransformer/ infra lifted (Tensor, types, config, loader, tokenizer, sampler), CUDA-stripped, smoke-tested.
- [ ] F5.2 — TernarySim transformer logic (our own attention/ffn/norm calling our kernels).
- [ ] F5.3 — TinyLlama smoke (load real weights, one forward pass).
```

Update the "Building blocks" table to note the new vendor section briefly.

- [ ] **Step 2: Create `docs/bridge-notes.md`**

```markdown
# `ter` Bridge Notes

Captured during F5.1 — lifting `~/ntransformer` infrastructure into `vendor/ntransformer/`.

## What we lifted

- `core/types.{h,cpp}` — `DType` enum, with `DType::TERNARY = 9` added.
- `core/device.h` — `Device::CPU` only.
- `core/tensor.{h,cpp}` — CUDA paths stripped, plain malloc.
- `model/config.{h,cpp}` — drop-in.
- `model/loader.{h,cpp}` — mmap GGUF parsing, CUDA-free.
- `inference/tokenizer.{h,cpp}` — drop-in (BPE, no CUDA).
- `inference/sampler.{h,cpp}` — drop-in (RNG only).

## What we did NOT lift

- `model/transformer.{h,cpp}`, `model/attention.{h,cpp}`, `model/ffn.{h,cpp}`, `model/norm.{h,cpp}` — these hardcode `cuda::launch_*()` calls; we'll write our own thin replacements that call our 5 ternary kernels (F5.2).
- `inference/engine.{h,cpp}` — entangled with the streaming pipeline; we'll write a minimal generation loop (F5.3).
- `src/cuda/*` — replaced entirely by our kernels and `Sim::call_kernel`.
- `src/memory/streamer.*` (NVMe streaming) — out of scope.

## Why this split

The exploration report found no backend abstraction in ntransformer. Patching attention.cpp / ffn.cpp / norm.cpp with `if (use_ternary_sim_)` branches at every call site would be invasive and ugly. Writing fresh transformer logic that calls our kernels directly is cleaner and gives us a clean Llama-on-ternary front-end without inheriting CUDA orchestration.

## Bridge contract (F5.2)

Our transformer code (in `src/host/transformer.cpp`, F5.2) will:
- Take an `nt::ModelConfig` and `nt::GGUFLoader` for setup.
- Use `nt::Tensor` for all activations and weights.
- Convert weights to `DType::TERNARY` at load time (per-tensor scale stored in a side struct).
- Call our 5 kernels (`tk_matmul_b_9t`, `tk_rmsnorm`, `tk_softmax`, `tk_silu`, `tk_rope`) for arithmetic.
- Use `nt::Tokenizer` for input/output and `nt::Sampler` for token selection.

This keeps the bridge surface small: only the transformer/attention/ffn/norm logic is new code; everything around it is the user's tested infrastructure.
```

- [ ] **Step 3: Build, verify (no test changes)**

Expected: 44/44 still.

- [ ] **Step 4: Commit**

```
git add README.md docs/bridge-notes.md
git commit -m "docs(ter): F5.1 done -- vendor lift summary + bridge notes"
```

---

## Self-Review

- **Spec coverage:** §9 vendor section partially covered (infrastructure lifted, transformer logic deferred to F5.2).
- **Placeholder scan:** all lifts are verbatim copies + minimal CUDA stripping. No TBDs.
- **Type consistency:** `nt::DType::TERNARY`, `nt::Tensor`, `nt::ModelConfig`, `nt::GGUFLoader`, `nt::Tokenizer`, `nt::Sampler` referenced consistently.
- **Known caveats:**
  - The synthetic GGUF in F5.1.5 may need iteration if the loader's parsing is strict about format details (alignment, magic, etc.). The fallback test (load nonexistent + load /etc/hosts) is the must-pass.
  - `core/allocator.{h,cpp}` may or may not be needed depending on whether `tensor.cpp` references it. The implementer decides at F5.1.3 time.

---

## Execution Handoff

After all tasks complete: `vendor/ntransformer/` exists with a `nt_infra` static library that compiles into `ter`. Smoke tests verify each module works in isolation. Tests count: 44/44.

Next plan: F5.2 — write `src/host/ter_transformer.cpp` that uses `nt::Tensor`/`nt::ModelConfig`/`nt::GGUFLoader` for setup, and our kernels for arithmetic, doing one full layer's forward pass.
