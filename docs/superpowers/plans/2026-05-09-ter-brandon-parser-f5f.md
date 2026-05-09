# `ter` Brandon GGUF Parser Plan (F5.4b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Extend the vendored `nt::GGUFLoader` to parse the 15 `brandon.*` metadata keys so `loader.config()` returns correct values for brandon-arch GGUFs. End state: opening `~/osito-a-models/build/brandon-tiny-10m-f16.gguf` produces a populated `BrandonConfig` (or extended `ModelConfig`) with `compute_layer_count=24`, `n_registers=4`, `use_dwa=true`, `use_value_residual=true`, `weight_tying=true`, `layer_map.size()=24`, plus the standard llama dims (dim=256, n_heads=8, n_kv_heads=2, etc.).

**Architecture:** Add a `BrandonConfig` struct alongside `ModelConfig` that holds the brandon-specific fields (logical layer count, layer_map array, registers, DWA, value_residual, weight tying). Extend the loader's metadata parser to recognize `brandon.*` keys and populate either `ModelConfig` (for fields that overlap with llama, like `embedding_length` → `hidden_size`) or `BrandonConfig` (for brandon-specific fields). When `general.architecture == "brandon"`, the loader populates both configs; for `llama` GGUFs only `ModelConfig` gets populated.

**Tech Stack:** Same as F5.4a.

**Spec / canonical reference:** `~/osito-k/docs/brandon-tiny-integration.md` Step 1 (the 15 key list with types). Memory: `ref_brandon_tiny_guide.md`.

**Out of scope:**
- SPM tokenizer (F5.4c — needs separate work).
- Tile-aware rmsnorm/softmax (F5.4d).
- Forward pass with brandon bits (F5.4e).
- Loading register/dwa weight tensors (deferred — fetched by name in F5.4e when needed).

---

## Source — the 15 `brandon.*` keys

Per the guide (verbatim):

```
brandon.embedding_length              # uint32, hidden dim (256)
brandon.block_count                   # uint32, *unique* blocks (12)
brandon.feed_forward_length           # uint32, SwiGLU intermediate (720)
brandon.attention.head_count          # uint32, total Q heads (8)
brandon.attention.head_count_kv       # uint32, GQA KV heads (2)
brandon.attention.layer_norm_rms_epsilon  # float32
brandon.context_length                # uint32, max_seq_len (512)
brandon.rope.freq_base                # float32, 10000.0
brandon.rope.dimension_count          # uint32, head_dim (32)
brandon.compute_layer_count           # uint32, logical layers (24); ≥ block_count
brandon.layer_map                     # int32 array[compute_layer_count]
brandon.use_dwa                       # bool
brandon.use_value_residual            # bool
brandon.n_registers                   # uint32 (4)
brandon.n_loops                       # uint32 (1 = no looping)
brandon.weight_tying                  # bool
```

(15 keys per the guide; `general.architecture == "brandon"` is a 16th key that already triggers the brandon path.)

The loader's existing pattern (per F5.1.5 lift) handles llama.* keys; extending it for brandon.* should follow the same pattern.

---

## File Structure (modifications)

```
vendor/ntransformer/model/
├── config.h            # add BrandonConfig struct + accessor on ModelConfig
└── loader.cpp          # extend metadata parser for brandon.* keys
tests/
└── test_brandon_config.cpp   # parse brandon-tiny GGUF, verify all 15 fields
```

---

## Task F5.4b.1 — `BrandonConfig` struct

**Files:**
- Modify `vendor/ntransformer/model/config.h`

**Critical git discipline:** Stay on `feature/f0-f2-foundation`, no detached HEAD.

- [ ] **Step 1: Extend `vendor/ntransformer/model/config.h`**

In the `nt` namespace, after the existing `ModelConfig` struct, add:

```cpp
// Brandon-specific metadata (parallel to ModelConfig for brandon-arch GGUFs).
// Populated when general.architecture == "brandon"; ignored for llama-arch.
struct BrandonConfig {
    int   block_count = 0;          // unique transformer blocks (12 for brandon-tiny)
    int   compute_layer_count = 0;  // logical layers (24)
    std::vector<int> layer_map;     // size = compute_layer_count; values in [0, block_count)
    int   n_registers = 0;          // learnable register tokens (4)
    int   n_loops = 1;              // 1 = no looping (only value used in current models)
    bool  use_dwa = false;
    bool  use_value_residual = false;
    bool  weight_tying = false;     // when true, output uses token_embd as LM head

    bool is_valid() const {
        // A populated BrandonConfig has block_count > 0 AND layer_map filled.
        return block_count > 0 && static_cast<int>(layer_map.size()) == compute_layer_count;
    }
};
```

Add `#include <vector>` at the top of config.h if not already present.

- [ ] **Step 2: Extend `ModelConfig` to optionally hold a `BrandonConfig`**

Inside `ModelConfig`, add a single member:

```cpp
    // Populated only for brandon-arch GGUFs. Empty/default-constructed otherwise.
    BrandonConfig brandon;
```

Place this just before the closing brace of `ModelConfig`.

- [ ] **Step 3: Build and verify**

```
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: 49/49 (header-only addition, no behavior change).

If any test breaks because `ModelConfig` is moved by value somewhere and the new `std::vector<int>` slows things down, that's a non-issue (we don't construct ModelConfig in hot paths).

- [ ] **Step 4: Commit**

```
git add vendor/ntransformer/model/config.h
git commit -m "feat(vendor): BrandonConfig struct for brandon-arch GGUF metadata"
```

After: `git log --oneline -2`.

---

## Task F5.4b.2 — Patch loader's metadata parser

**Files:**
- Modify `vendor/ntransformer/model/loader.cpp`
- (Possibly modify `vendor/ntransformer/model/config.cpp` if `from_gguf_metadata()` lives there)

The loader's existing parser walks GGUF metadata key-value pairs. Find the function (likely `ModelConfig::from_gguf_metadata()` per the F5.1 lift) and add brandon.* handlers parallel to the llama.* ones.

- [ ] **Step 1: Read the existing metadata parser**

Open both `vendor/ntransformer/model/loader.cpp` AND `vendor/ntransformer/model/config.cpp` (the F5.1 lift may have either). Find where `llama.embedding_length`, `llama.block_count`, etc. are parsed. Note the helper functions used (e.g. `get_kv_uint32(name)`, `get_kv_string(name)`, `get_kv_array(name)`).

- [ ] **Step 2: Add brandon.* parsing**

After the existing llama.* parsing block, add a parallel brandon.* block. Pattern:

```cpp
const auto& arch = get_kv_string("general.architecture");
if (arch == "brandon") {
    // Standard dims (overlap with llama, just brandon.* prefix)
    hidden_size       = get_kv_uint32("brandon.embedding_length",         hidden_size);
    intermediate_size = get_kv_uint32("brandon.feed_forward_length",      intermediate_size);
    n_heads           = get_kv_uint32("brandon.attention.head_count",     n_heads);
    n_kv_heads        = get_kv_uint32("brandon.attention.head_count_kv",  n_kv_heads);
    head_dim          = get_kv_uint32("brandon.rope.dimension_count",     head_dim);
    norm_eps          = get_kv_float ("brandon.attention.layer_norm_rms_epsilon", norm_eps);
    max_seq_len       = get_kv_uint32("brandon.context_length",           max_seq_len);
    rope_theta        = get_kv_float ("brandon.rope.freq_base",           rope_theta);

    // Brandon-specific (populate the nested struct)
    brandon.block_count         = get_kv_uint32("brandon.block_count",         0);
    brandon.compute_layer_count = get_kv_uint32("brandon.compute_layer_count", 0);
    brandon.n_registers         = get_kv_uint32("brandon.n_registers",         0);
    brandon.n_loops             = get_kv_uint32("brandon.n_loops",             1);
    brandon.use_dwa             = get_kv_bool  ("brandon.use_dwa",             false);
    brandon.use_value_residual  = get_kv_bool  ("brandon.use_value_residual",  false);
    brandon.weight_tying        = get_kv_bool  ("brandon.weight_tying",        false);

    // layer_map is an int32 array of length compute_layer_count
    brandon.layer_map = get_kv_int32_array("brandon.layer_map");

    // Make ModelConfig::n_layers consistent with the logical layer count for the forward loop.
    n_layers = brandon.compute_layer_count;
}
```

Adjust accessor names to match what the existing parser uses (the helpers may be named differently — `gguf_get_u32(...)`, `gguf_get_arr_i32(...)`, etc.). If `get_kv_bool` doesn't exist, use `get_kv_uint8` or `get_kv_uint32` and cast.

If `get_kv_int32_array` doesn't exist either, write a one-off helper next to the existing accessors. The GGUF format for typed arrays is: 4 bytes type tag + 8 bytes length + N elements. The existing parser surely has a generic array reader.

- [ ] **Step 3: Verify the existing parser uses an "if architecture == X" guard**

If the parser uses a string-switch over architectures (e.g., `if (arch == "llama") {...}`), add `else if (arch == "brandon") {...}` next to it. If it just runs llama.* unconditionally, add the brandon.* block in parallel — the `get_kv_*` helpers should return defaults when the key is absent, so no crash.

- [ ] **Step 4: Build (no new test yet)**

```
cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: 49/49 still. The brandon test from F5.4a still skips on missing keys because `loader.config().n_layers` was 0 before this change too — but now it should populate to 24 if the GGUF is present.

- [ ] **Step 5: Commit**

```
git add vendor/ntransformer/model/loader.cpp vendor/ntransformer/model/config.cpp
git commit -m "feat(vendor): parse brandon.* GGUF metadata into BrandonConfig"
```

After: `git log --oneline -2`.

---

## Task F5.4b.3 — End-to-end test: parse brandon-tiny GGUF and verify fields

**Files:**
- Create `tests/test_brandon_config.cpp`
- Modify `tests/CMakeLists.txt`

- [ ] **Step 1: Write `tests/test_brandon_config.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <model/loader.h>
#include <fstream>

using namespace nt;

namespace {
constexpr const char* GGUF_PATH = "/Users/pc/osito-a-models/build/brandon-tiny-10m-f16.gguf";

bool file_exists(const char* path) {
    std::ifstream f(path);
    return f.good();
}
}

TEST_CASE("brandon-tiny config populates BrandonConfig correctly") {
    if (!file_exists(GGUF_PATH)) {
        MESSAGE("brandon-tiny GGUF not found — skipping");
        return;
    }

    GGUFLoader loader;
    REQUIRE(loader.load(GGUF_PATH));

    const auto& cfg = loader.config();

    // Standard dims (now correctly populated from brandon.* keys).
    CHECK(cfg.hidden_size       == 256);
    CHECK(cfg.intermediate_size == 720);
    CHECK(cfg.n_heads           == 8);
    CHECK(cfg.n_kv_heads        == 2);
    CHECK(cfg.head_dim          == 32);
    CHECK(cfg.max_seq_len       == 512);
    CHECK(cfg.norm_eps          == doctest::Approx(1e-5f).epsilon(1e-3));
    CHECK(cfg.rope_theta        == doctest::Approx(10000.0f));

    // Brandon-specific.
    const auto& b = cfg.brandon;
    CHECK(b.block_count         == 12);
    CHECK(b.compute_layer_count == 24);
    CHECK(b.n_registers         == 4);
    CHECK(b.n_loops             == 1);
    CHECK(b.use_dwa             == true);
    CHECK(b.use_value_residual  == true);
    CHECK(b.weight_tying        == true);

    REQUIRE(b.layer_map.size() == 24);
    // Each entry must be a valid block index in [0, block_count).
    for (auto idx : b.layer_map) {
        CHECK(idx >= 0);
        CHECK(idx < b.block_count);
    }

    // n_layers should be compute_layer_count for forward-loop iteration.
    CHECK(cfg.n_layers == 24);

    CHECK(b.is_valid());
}
```

- [ ] **Step 2: Wire test in `tests/CMakeLists.txt`**

```cmake
ter_add_test(test_brandon_config)
target_link_libraries(test_brandon_config PRIVATE nt_infra)
```

- [ ] **Step 3: Build and run**

Expected: 50/50 (49 prior + new). Test SKIPS gracefully if the GGUF is absent.

If a CHECK fails, debug:
- The expected values come from the brandon-tiny config.json (`hidden=256, n_layers=24, n_heads=8, n_kv_heads=2, vocab=8192, hidden_dim=720, max_seq_len=512`). Mismatches mean the parser's key name is off or the GGUF stores a different value.
- If `b.layer_map` is empty: the array reader path may not be wired. Inspect with debug prints before fixing.
- If `b.use_dwa` is false but should be true: the bool decoder path may need adjustment (GGUF bool is sometimes 1-byte uint8).

- [ ] **Step 4: Commit**

```
git add tests/test_brandon_config.cpp tests/CMakeLists.txt
git commit -m "test(vendor): F5.4b — brandon-tiny GGUF parses all 15 brandon.* fields"
```

After: `git log --oneline -3`.

---

## Final Task — README + bridge notes

- [ ] **Step 1: Update `README.md` Status block**

Replace the F5.4b line with:
```markdown
- [x] F5.4b — Extend GGUF parser for `brandon.*` keys; BrandonConfig populated correctly from real brandon-tiny GGUF.
```

- [ ] **Step 2: Append to `docs/bridge-notes.md`**

```markdown
## F5.4b result — brandon.* GGUF parser

The vendored `nt::ModelConfig` now holds a `BrandonConfig` substruct populated only when `general.architecture == "brandon"`. All 15 brandon.* keys from `~/osito-k/docs/brandon-tiny-integration.md` Step 1 are recognized. Test verifies every field against the brandon-tiny-10m-instruct config:

- Standard dims via brandon.* aliases: hidden=256, n_heads=8, n_kv_heads=2, head_dim=32, intermediate=720, max_seq=512.
- Brandon-specific: block_count=12, compute_layer_count=24, n_registers=4, use_dwa=true, use_value_residual=true, weight_tying=true.
- `layer_map` is a 24-entry int32 array with each value in [0, 12).

`ModelConfig::n_layers` is overloaded to mean "logical layer count for the forward loop". For brandon, this is `compute_layer_count` (24). For llama, it stays as `llama.block_count`.

This unblocks F5.4d (multi-layer forward with brandon bits) — we now have the metadata to drive layer_map fanout, register prefill, DWA mixing, and value_residual.
```

- [ ] **Step 3: Build, verify**

Expected: 50/50.

- [ ] **Step 4: Commit**

```
git add README.md docs/bridge-notes.md
git commit -m "docs(ter): F5.4b done -- brandon.* parser; metadata ready for forward pass"
```

---

## Self-Review

- **Spec coverage:** Step 1 of `~/osito-k/docs/brandon-tiny-integration.md` covered. Steps 2-10 follow in F5.4c-f.
- **Placeholder scan:** the `get_kv_*` accessor names are placeholders — the implementer must adjust to match the actual loader pattern.
- **Type consistency:** `BrandonConfig`, `ModelConfig::brandon`, accessor field names referenced consistently.
- **Known caveats:** the bool decode (use_dwa, use_value_residual, weight_tying) may need adjustment for GGUF's actual bool encoding. The brandon test will tell us if this fails.

---

## Execution Handoff

After all tasks: `loader.config().brandon` is populated for brandon GGUFs. `loader.config().n_layers == 24`. We have all the metadata needed for the forward-pass plumbing in F5.4d (layer_map fanout, register count, DWA flag, value_residual flag, weight tying).

Next plan: F5.4c — SPM tokenizer (vendored ntransformer's tokenizer is BPE-only; brandon needs SentencePiece per the integration guide Step 5). After that: F5.4d (forward pass with brandon bits), F5.4e (sampling recipe + ChatML), F5.4f (calibrate "Who was Einstein?").
