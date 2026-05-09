# `ter` Multi-Token + KV Cache Plan (F5.3a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Extend `forward_layer()` to handle multi-token sequential decoding with a per-layer KV cache. End state: calling `forward_layer()` repeatedly with positions 0, 1, 2, 3 produces the same output as a numpy reference doing causal multi-token attention. RMSNorm/RoPE/Softmax/SiLU stay host-side; F5.3b plumbs them into kernels.

**Architecture:** Add a `LayerActivations` struct that holds the per-layer KV cache as plain `std::vector<float>` (max_seq × n_kv_heads × head_dim). On each forward call, the current K/V get written into the cache at `pos`, and attention reads `K_cache[0..pos+1]` and `V_cache[0..pos+1]` to compute causal attention. Softmax is over a `pos+1`-length score vector per head (host-side for now).

The K/V vectors are also rotated with RoPE before being cached, per Llama spec.

**Tech Stack:** Same as F5.2.

**Spec:** `docs/superpowers/specs/2026-05-08-ter-design.md` §9. Bridge contract in `docs/bridge-notes.md`.

**Out of scope:**
- Plumbing tk_rmsnorm/tk_softmax/tk_silu/tk_rope into `forward_layer` — F5.3b. We keep these host-side.
- Multi-layer chaining — mechanical, lands when needed.
- GGUF weight loading — F5.4.
- TinyLlama or Llama 3.2 1B end-to-end — F5.4 / F6.

---

## Why split F5.3 into a/b

Multi-token + KV cache is an algorithmic change (the structure of attention changes). Plumbing the transcendental kernels into forward_layer is a mechanical change (swap one path for another). Doing both in one plan inflates the test surface and confuses bug attribution. Split keeps each step's diff focused.

---

## File Structure

```
include/ter/tx/
├── layer.hpp              # extend with LayerActivations / KVCache struct
└── forward.hpp            # extend forward_layer signature with cache + max_seq
src/tx/
└── forward.cpp            # update for multi-token attention with cache
tests/
└── test_layer_multitoken.cpp   # 4 sequential forward calls vs numpy reference
```

No new files for the kernel-plumbing — that's F5.3b.

---

## Task F5.3a.1 — `LayerActivations` / `KVCache` struct

**Files:**
- Modify `include/ter/tx/layer.hpp` — add `KVCache` struct
- Modify `include/ter/tx/forward.hpp` — add `LayerActivations` (just the cache for now)

The cache layout is plain row-major `[max_seq, n_kv_heads * head_dim]` for K and same for V.

- [ ] **Step 1: Append to `include/ter/tx/layer.hpp`** (after the existing struct/decls):

```cpp
// Per-layer scratch state that persists across token positions.
struct KVCache {
    int max_seq = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    std::vector<float> K;   // (max_seq, n_kv_heads * head_dim)
    std::vector<float> V;   // (max_seq, n_kv_heads * head_dim)

    void resize(int new_max_seq, int new_n_kv_heads, int new_head_dim) {
        max_seq = new_max_seq;
        n_kv_heads = new_n_kv_heads;
        head_dim = new_head_dim;
        size_t n = static_cast<size_t>(max_seq) *
                   static_cast<size_t>(n_kv_heads) *
                   static_cast<size_t>(head_dim);
        K.assign(n, 0.0f);
        V.assign(n, 0.0f);
    }

    int kv_dim() const { return n_kv_heads * head_dim; }
};
```

- [ ] **Step 2: Verify build still passes**

```
cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: 46/46 still (header-only addition, no behavior change yet).

- [ ] **Step 3: Commit**

```
git add include/ter/tx/layer.hpp
git commit -m "feat(tx): KVCache struct for per-layer state"
```

After: `git log --oneline -2`.

---

## Task F5.3a.2 — Update `forward_layer()` signature for multi-token

**Files:**
- Modify `include/ter/tx/forward.hpp`
- Modify `src/tx/forward.cpp`

The new signature takes a `KVCache&` and updates it in place. Attention reads cache[0..pos+1] for causal scope.

- [ ] **Step 1: Update `include/ter/tx/forward.hpp`**

Replace the `forward_layer` declaration with:
```cpp
void forward_layer(
    Sim& sim,
    KernelTable& kt,
    const LayerWeights& L,
    KVCache& cache,                          // NEW: in/out per-layer state
    const std::vector<float>& hidden_in,
    int pos,
    int hidden_size,
    int head_dim,
    int n_heads,
    int n_kv_heads,
    int intermediate_size,
    float rmsnorm_eps,
    const LutAddrs& luts,
    std::vector<float>& hidden_out);
```

Add `#include <ter/tx/layer.hpp>` at the top if not already there (LayerActivations / KVCache live in layer.hpp).

- [ ] **Step 2: Update `src/tx/forward.cpp`**

Replace the body of `forward_layer()` with the new multi-token version. The differences from F5.2:

1. After computing K and V (with RoPE applied), write into `cache.K[pos]` and `cache.V[pos]`.
2. Before computing context, materialise `K_view = cache.K[0..pos+1]` and `V_view = cache.V[0..pos+1]`.
3. Per query head, compute attention scores against all past key positions for the corresponding KV head.
4. Softmax over `pos+1` scores.
5. Weighted sum over V.

Replace the relevant section (after RoPE on Q/K, before Wo) with:

```cpp
    // 3.5) Cache K and V at position `pos`.
    int kv_dim = n_kv_heads * head_dim;
    REQUIRE_FALSE(false);  // sanity placeholder — remove if implementing cleanly
    for (int i = 0; i < kv_dim; ++i) {
        cache.K[static_cast<size_t>(pos) * kv_dim + i] = k[i];
        cache.V[static_cast<size_t>(pos) * kv_dim + i] = v[i];
    }

    // 4) Causal attention over [0..pos].
    //    For each query head h: scores[t] = Q[h] · K_cache[t][kv_head_of(h)] / sqrt(head_dim)
    //    Then softmax and weighted sum over V_cache[0..pos+1][kv_head_of(h)].
    int q_dim = n_heads * head_dim;
    int gqa_group = n_heads / n_kv_heads;   // 1 if not GQA
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> ctx(q_dim, 0.0f);
    for (int h = 0; h < n_heads; ++h) {
        int kv_h = h / gqa_group;
        const float* qh = q.data() + h * head_dim;

        // scores[t] for t in [0, pos+1]
        std::vector<float> scores(pos + 1, 0.0f);
        for (int t = 0; t <= pos; ++t) {
            const float* kh = cache.K.data() + (static_cast<size_t>(t) * kv_dim) +
                              static_cast<size_t>(kv_h) * head_dim;
            double acc = 0.0;
            for (int d = 0; d < head_dim; ++d) acc += double(qh[d]) * double(kh[d]);
            scores[t] = static_cast<float>(acc) * inv_sqrt_d;
        }

        // softmax
        double mx = scores[0];
        for (auto s_ : scores) if (s_ > mx) mx = s_;
        double sum = 0.0;
        for (auto& s_ : scores) { s_ = static_cast<float>(std::exp(double(s_) - mx)); sum += s_; }
        for (auto& s_ : scores) s_ = static_cast<float>(s_ / sum);

        // weighted sum over V
        float* ch = ctx.data() + h * head_dim;
        for (int t = 0; t <= pos; ++t) {
            const float* vh = cache.V.data() + (static_cast<size_t>(t) * kv_dim) +
                              static_cast<size_t>(kv_h) * head_dim;
            for (int d = 0; d < head_dim; ++d) ch[d] += scores[t] * vh[d];
        }
    }
```

(Strip the `REQUIRE_FALSE` placeholder; that's just to flag the spot during editing.)

The rest of the function (Wo projection, residual, ffn_norm, gate/up, SwiGLU, Wdown, residual) is unchanged from F5.2.

- [ ] **Step 3: Build (no new test yet)**

```
cmake --build build && ctest --test-dir build --output-on-failure
```

The existing `test_layer_forward` from F5.2 will FAIL because it calls `forward_layer` with the old signature. Update it to construct an empty KVCache and pass it. Single-token (pos=0) behavior is identical to F5.2 once the cache stores K/V at index 0 and attention is just one position deep.

Update `tests/test_layer_forward.cpp`:
- After `LayerWeights L = ...`:
```cpp
KVCache cache;
cache.resize(/*max_seq*/8, Kn, HD);
```
- Update the `forward_layer(...)` call to pass `cache` after `L`.

After this, expected: 46/46 still (test_layer_forward should still pass at pos=0 with empty cache).

- [ ] **Step 4: Commit**

```
git add include/ter/tx/forward.hpp src/tx/forward.cpp tests/test_layer_forward.cpp
git commit -m "feat(tx): multi-token attention with KV cache (host-side softmax)"
```

After: `git log --oneline -2`.

---

## Task F5.3a.3 — Multi-token end-to-end test

**Files:**
- Create `tests/test_layer_multitoken.cpp`
- Modify `tests/CMakeLists.txt`

The test:
1. Same random Layer weights as F5.2.
2. 4 sequential `forward_layer()` calls with positions 0, 1, 2, 3, sharing the same KVCache.
3. Reference: a numpy-equivalent multi-token causal attention computed for the same hidden_in sequence.
4. Compare outputs at position 3 (the last) within bounded rel_err.

- [ ] **Step 1: Write `tests/test_layer_multitoken.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/tx/forward.hpp>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#ifndef TER_KERNELS_DIR
#error "TER_KERNELS_DIR must be defined"
#endif

using namespace ter;
using namespace ter::tx;

namespace {

constexpr int H  = 4;
constexpr int HD = 4;
constexpr int Hn = 1, Kn = 1;
constexpr int I  = 8;
constexpr int SEQ = 4;

// Reference: numpy-equivalent multi-token forward over SEQ positions.
// Returns the SEQ×H output matrix (one row per position).
std::vector<std::vector<float>> numpy_multitoken_forward(
    const std::vector<std::vector<float>>& hidden_in_seq,
    const std::vector<float>& Wq, const std::vector<float>& Wk,
    const std::vector<float>& Wv, const std::vector<float>& Wo,
    const std::vector<float>& Wgate, const std::vector<float>& Wup,
    const std::vector<float>& Wdown,
    const std::vector<float>& nw1, const std::vector<float>& nw2)
{
    auto matvec = [](const std::vector<float>& x, const std::vector<float>& W,
                     int K, int N) {
        std::vector<float> y(N, 0.0f);
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < K; ++k) y[j] += x[k] * W[k * N + j];
        return y;
    };
    auto rmsnorm = [](const std::vector<float>& x, const std::vector<float>& w, float eps) {
        double ss = 0.0;
        for (auto v : x) ss += double(v) * double(v);
        double inv = 1.0 / std::sqrt(ss / x.size() + eps);
        std::vector<float> y(x.size());
        for (size_t i = 0; i < x.size(); ++i) y[i] = static_cast<float>(x[i] * inv) * w[i];
        return y;
    };
    auto rope = [](std::vector<float>& v, int pos, int head_dim) {
        for (int k = 0; k < head_dim / 2; ++k) {
            double freq = 1.0 / std::pow(10000.0, (2.0 * k) / double(head_dim));
            double angle = double(pos) * freq;
            double c = std::cos(angle), s = std::sin(angle);
            float x0 = v[2 * k], x1 = v[2 * k + 1];
            v[2 * k]     = static_cast<float>(x0 * c - x1 * s);
            v[2 * k + 1] = static_cast<float>(x0 * s + x1 * c);
        }
    };

    // Allocate KV cache.
    std::vector<std::vector<float>> K_cache(SEQ, std::vector<float>(Kn * HD, 0.0f));
    std::vector<std::vector<float>> V_cache(SEQ, std::vector<float>(Kn * HD, 0.0f));

    std::vector<std::vector<float>> outs(SEQ);

    for (int pos = 0; pos < SEQ; ++pos) {
        const auto& hidden_in = hidden_in_seq[pos];

        auto x_norm = rmsnorm(hidden_in, nw1, 1e-6f);
        auto q = matvec(x_norm, Wq, H, HD);
        auto k = matvec(x_norm, Wk, H, HD);
        auto v = matvec(x_norm, Wv, H, HD);

        rope(q, pos, HD);
        rope(k, pos, HD);

        K_cache[pos] = k;
        V_cache[pos] = v;

        // Causal attention at this position.
        std::vector<float> ctx(HD, 0.0f);
        float inv_sd = 1.0f / std::sqrt(float(HD));
        std::vector<float> scores(pos + 1, 0.0f);
        for (int t = 0; t <= pos; ++t) {
            double acc = 0.0;
            for (int d = 0; d < HD; ++d) acc += double(q[d]) * double(K_cache[t][d]);
            scores[t] = static_cast<float>(acc) * inv_sd;
        }
        double mx = scores[0];
        for (auto s_ : scores) if (s_ > mx) mx = s_;
        double sum = 0.0;
        for (auto& s_ : scores) { s_ = static_cast<float>(std::exp(double(s_) - mx)); sum += s_; }
        for (auto& s_ : scores) s_ = static_cast<float>(s_ / sum);
        for (int t = 0; t <= pos; ++t) {
            for (int d = 0; d < HD; ++d) ctx[d] += scores[t] * V_cache[t][d];
        }

        auto attn_out = matvec(ctx, Wo, HD, H);

        std::vector<float> mid(H);
        for (int i = 0; i < H; ++i) mid[i] = hidden_in[i] + attn_out[i];

        auto mid_norm = rmsnorm(mid, nw2, 1e-6f);
        auto gate = matvec(mid_norm, Wgate, H, I);
        auto up   = matvec(mid_norm, Wup,   H, I);

        std::vector<float> ff(I);
        for (int i = 0; i < I; ++i) {
            double s = 1.0 / (1.0 + std::exp(-double(gate[i])));
            ff[i] = static_cast<float>(double(gate[i]) * s * double(up[i]));
        }

        auto ff_out = matvec(ff, Wdown, I, H);

        std::vector<float> out(H);
        for (int i = 0; i < H; ++i) out[i] = mid[i] + ff_out[i];
        outs[pos] = out;
    }

    return outs;
}

}  // namespace

TEST_CASE("forward_layer multi-token causal attention matches numpy") {
    std::mt19937 rng(0xCABA);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    // Random sequence of SEQ hidden_in vectors.
    std::vector<std::vector<float>> hidden_in_seq(SEQ, std::vector<float>(H));
    for (auto& row : hidden_in_seq) for (auto& v : row) v = dist(rng);

    std::vector<float> Wq(H * HD), Wk(H * HD), Wv(H * HD), Wo(HD * H);
    std::vector<float> Wgate(H * I), Wup(H * I), Wdown(I * H);
    std::vector<float> nw1(H, 1.0f), nw2(H, 1.0f);
    for (auto& v : Wq)    v = dist(rng);
    for (auto& v : Wk)    v = dist(rng);
    for (auto& v : Wv)    v = dist(rng);
    for (auto& v : Wo)    v = dist(rng);
    for (auto& v : Wgate) v = dist(rng);
    for (auto& v : Wup)   v = dist(rng);
    for (auto& v : Wdown) v = dist(rng);

    // Reference.
    auto outs_ref = numpy_multitoken_forward(hidden_in_seq,
        Wq, Wk, Wv, Wo, Wgate, Wup, Wdown, nw1, nw2);

    // Quantize and set up sim.
    LayerWeights L = quantize_layer(
        Wq.data(), H, HD,
        Wk.data(), H, HD,
        Wv.data(), H, HD,
        Wo.data(), HD, H,
        Wgate.data(), H, I,
        Wup.data(),   H, I,
        Wdown.data(), I, H,
        nw1.data(), H,
        nw2.data(), H);

    Sim s(8 * 1024);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);

    KVCache cache;
    cache.resize(SEQ, Kn, HD);

    LutAddrs luts{0, 0, 0, 0};

    // Run SEQ sequential forward_layer calls.
    std::vector<std::vector<float>> outs(SEQ);
    for (int pos = 0; pos < SEQ; ++pos) {
        forward_layer(s, kt, L, cache, hidden_in_seq[pos], pos,
                      H, HD, Hn, Kn, I, 1e-6f, luts, outs[pos]);
    }

    // Compare every position's output.
    double max_rel = 0.0;
    for (int pos = 0; pos < SEQ; ++pos) {
        REQUIRE(outs[pos].size() == static_cast<size_t>(H));
        for (int i = 0; i < H; ++i) {
            double ref = outs_ref[pos][i];
            double got = outs[pos][i];
            double denom = std::max(0.1, std::fabs(ref));
            double rel = std::fabs(got - ref) / denom;
            if (rel > max_rel) max_rel = rel;
        }
    }
    CHECK(max_rel < 0.5);

    // Counter sanity.
    CHECK(s.counters().get(Opcode::TVMAC) > 0);
}
```

- [ ] **Step 2: Wire test in `tests/CMakeLists.txt`**

```cmake
ter_add_test(test_layer_multitoken)
target_compile_definitions(test_layer_multitoken PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
```

- [ ] **Step 3: Build and run**

Expected: 47/47.

If `max_rel >= 0.5`:
- Common bugs:
  - K/V being written to cache BEFORE rope (should be after, per the kernel signature; the code in F5.3a.2 writes after rope already)
  - Index math in `K_cache[t][kv_h]` — verify with hand check
  - GQA grouping: with Hn=Kn=1, `gqa_group = 1`, so `kv_h = h`. Verify.
  - Causal scope: `scores` should have length `pos+1`, not `pos`. Off-by-one.
- Print outs[pos] vs outs_ref[pos] for each position to localize.

If max_rel ~5-30%, accept.

- [ ] **Step 4: Commit**

```
git add tests/test_layer_multitoken.cpp tests/CMakeLists.txt
git commit -m "test(tx): F5.3a — multi-token causal attention via KV cache vs numpy"
```

After: `git log --oneline -3`.

---

## Final Task — README + bridge notes update

**Files:**
- Modify `README.md` Status block
- Append to `docs/bridge-notes.md`

- [ ] **Step 1: Update `README.md`**

Replace the F5.3 line with:
```markdown
- [x] F5.3a — Multi-token attention with KV cache; sequential forward_layer over 4 positions matches numpy.
- [ ] F5.3b — Plumb tk_rmsnorm/tk_softmax/tk_silu/tk_rope into forward_layer (replace host-side stubs).
```

- [ ] **Step 2: Append to `docs/bridge-notes.md`**

```markdown
## F5.3a result — multi-token + KV cache

- Added `ter::tx::KVCache` struct: per-layer K/V tensors of shape (max_seq, n_kv_heads * head_dim), in plain `std::vector<float>` (host-side).
- `forward_layer()` now writes the current K/V into the cache at `pos` and computes causal attention over `[0..pos+1]`.
- Test: 4 sequential `forward_layer()` calls with shared cache match the numpy multi-token reference within bounded rel_err on H=4, HD=4, I=8, SEQ=4.
- Matmul still routes through `tk_matmul_b_9t`; everything else (RMSNorm, RoPE, Softmax, SiLU) is host-side. F5.3b plumbs the kernels.

### KV cache lives in host memory, not sim memory

The cache could live in sim memory (one big region), but for now it stays as `std::vector<float>` on the host because:
1. Attention reads the cache via host orchestration (one matmul per query position).
2. Sim memory is the kernel's working set; cache writes/reads from sim would add DMA overhead with no gain.
3. Future K2 (sim-resident transformer) will move the cache into sim memory.
```

- [ ] **Step 3: Build, verify**

Expected: 47/47 still.

- [ ] **Step 4: Commit**

```
git add README.md docs/bridge-notes.md
git commit -m "docs(ter): F5.3a done -- multi-token + KV cache"
```

After: `git log --oneline -3`.

---

## Self-Review

- **Spec coverage:** §9 — partial. F5.3a covers the multi-token attention algorithm; F5.3b covers the kernel plumbing.
- **Placeholder scan:** the existing `test_layer_forward` test gets adapted to pass an empty KVCache; that's a real change, not a placeholder.
- **Type consistency:** `KVCache`, `LutAddrs`, `forward_layer`, `quantize_layer` referenced consistently.
- **Known caveats:**
  - No GQA shape variation tested (Kn=Hn=1). Multi-head + GQA test would need adjusting `gqa_group` math.
  - `pos` is bounded by `cache.max_seq`. We assume the caller doesn't exceed it.

---

## Execution Handoff

After all tasks: `forward_layer()` does correct multi-token attention with persistent KV cache. The transformer flow is now algorithmically complete (just missing kernel plumbing for transcendentals).

Next plan (F5.3b): plumb `tk_rmsnorm`, `tk_softmax`, `tk_silu`, `tk_rope` into `forward_layer`, replacing the host-side stubs. After F5.3b, every arithmetic op except the host bookkeeping happens in a kernel.
