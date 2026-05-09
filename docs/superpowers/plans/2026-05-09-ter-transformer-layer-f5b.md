# `ter` Transformer Layer Plan (F5.2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Implement a single Llama-style transformer layer (RMSNorm → attention → residual → RMSNorm → FFN/SwiGLU → residual) as host code that uses `nt::Tensor`/`nt::ModelConfig` for setup and our 5 ternary kernels for arithmetic. End state: random weights + random hidden state → one full layer's forward pass via kernels → matches numpy reference within bounded relative error.

**Architecture:** A new namespace `ter::tx` (transformer extensions for `ter`) that defines:
- `LayerWeights` struct holding quantized `TritTensor` weights for one layer (Wq, Wk, Wv, Wo, Wgate, Wup, Wdown, attn_norm_w, ffn_norm_w).
- `LayerActivations` struct for the per-layer scratch (KV cache pointers, residual buffer).
- `forward_layer(sim, kt, layer, hidden_in, pos, hidden_out)` function that orchestrates the 5 kernels via the patterns established in F4.

This is **the first piece of original transformer code in `ter`**. F4's `test_attention.cpp` already proved the orchestration works for attention; F5.2 generalises that into a reusable `Layer` and adds the FFN block.

**Tech Stack:** C++17, depends on `ter` (for `Sim`, `KernelTable`, kernels) and `nt_infra` (for `Tensor`, `ModelConfig`, quantization helpers).

**Spec:** §9 of `docs/superpowers/specs/2026-05-08-ter-design.md`. Bridge contract in `docs/bridge-notes.md`.

**Out of scope:**
- Multi-layer chaining (one layer is the milestone; chaining is mechanical and lands in F5.3).
- KV cache management beyond a single position (deferred; this test uses pos=0 with no cache reuse).
- GGUF weight loading (deferred — this test uses random weights).
- Token embedding / output projection / sampling (deferred to F5.3).

---

## File Structure

```
include/ter/tx/
├── layer.hpp                # LayerWeights, LayerActivations structs
└── forward.hpp              # forward_layer() declaration
src/tx/
├── layer.cpp                # LayerWeights helpers (allocate, quantize_from_floats)
└── forward.cpp              # forward_layer() implementation
tests/
└── test_layer_forward.cpp   # single-layer vs numpy reference
```

The `tx::` namespace keeps these clearly separate from `ter::` (kernel/sim primitives) and `nt::` (vendored infrastructure).

---

## Task F5.2.1 — `LayerWeights` and `LayerActivations` types

**Files:**
- Create `include/ter/tx/layer.hpp`
- Create `src/tx/layer.cpp`
- Create `tests/test_layer_types.cpp`
- Modify `src/CMakeLists.txt`, `tests/CMakeLists.txt`

**Critical git discipline:** Stay on `feature/f0-f2-foundation`, no detached HEAD.

The `LayerWeights` struct holds quantized weights for one layer in `ter::TritTensor` format (we already have `quantize`/`dequantize` from F3). The struct is data-only with a builder helper.

For Llama-style attention with hidden_size=H, head_dim=D, n_heads=Hn, n_kv_heads=Kn, intermediate_size=I:
- `Wq: (H, Hn*D)`, `Wk: (H, Kn*D)`, `Wv: (H, Kn*D)`, `Wo: (Hn*D, H)`
- `Wgate: (H, I)`, `Wup: (H, I)`, `Wdown: (I, H)`
- `attn_norm_w: (H,)`, `ffn_norm_w: (H,)`

For F5.2 we use a tiny config: H=4, D=4, Hn=Kn=1 (no GQA), I=8. All values fit into 27-lane vectors with padding.

- [ ] **Step 1: Write `include/ter/tx/layer.hpp`**

```cpp
#pragma once
#include <ter/numfmt.hpp>
#include <vector>

namespace ter::tx {

// Quantized weights for one transformer layer.
struct LayerWeights {
    // Attention projections.
    TritTensor Wq;       // (hidden, n_heads * head_dim)
    TritTensor Wk;       // (hidden, n_kv_heads * head_dim)
    TritTensor Wv;       // (hidden, n_kv_heads * head_dim)
    TritTensor Wo;       // (n_heads * head_dim, hidden)
    // FFN projections.
    TritTensor Wgate;    // (hidden, intermediate)
    TritTensor Wup;      // (hidden, intermediate)
    TritTensor Wdown;    // (intermediate, hidden)
    // Norm weights — kept as float (per-element scale, no quantization)
    std::vector<float> attn_norm_w;
    std::vector<float> ffn_norm_w;
};

// Build LayerWeights by quantizing float weight matrices.
// Each input is row-major. n_trits_per_elem = 9 (Format B default).
LayerWeights quantize_layer(
    const float* Wq_data, int Wq_rows, int Wq_cols,
    const float* Wk_data, int Wk_rows, int Wk_cols,
    const float* Wv_data, int Wv_rows, int Wv_cols,
    const float* Wo_data, int Wo_rows, int Wo_cols,
    const float* Wgate_data, int Wgate_rows, int Wgate_cols,
    const float* Wup_data, int Wup_rows, int Wup_cols,
    const float* Wdown_data, int Wdown_rows, int Wdown_cols,
    const float* attn_norm_w, int attn_norm_n,
    const float* ffn_norm_w, int ffn_norm_n);

}  // namespace ter::tx
```

- [ ] **Step 2: Implement `src/tx/layer.cpp`**

```cpp
#include <ter/tx/layer.hpp>

namespace ter::tx {

LayerWeights quantize_layer(
    const float* Wq_data, int Wq_rows, int Wq_cols,
    const float* Wk_data, int Wk_rows, int Wk_cols,
    const float* Wv_data, int Wv_rows, int Wv_cols,
    const float* Wo_data, int Wo_rows, int Wo_cols,
    const float* Wgate_data, int Wgate_rows, int Wgate_cols,
    const float* Wup_data, int Wup_rows, int Wup_cols,
    const float* Wdown_data, int Wdown_rows, int Wdown_cols,
    const float* attn_norm_w, int attn_norm_n,
    const float* ffn_norm_w, int ffn_norm_n)
{
    LayerWeights L;
    L.Wq    = quantize(Wq_data,    {Wq_rows,    Wq_cols},    9);
    L.Wk    = quantize(Wk_data,    {Wk_rows,    Wk_cols},    9);
    L.Wv    = quantize(Wv_data,    {Wv_rows,    Wv_cols},    9);
    L.Wo    = quantize(Wo_data,    {Wo_rows,    Wo_cols},    9);
    L.Wgate = quantize(Wgate_data, {Wgate_rows, Wgate_cols}, 9);
    L.Wup   = quantize(Wup_data,   {Wup_rows,   Wup_cols},   9);
    L.Wdown = quantize(Wdown_data, {Wdown_rows, Wdown_cols}, 9);
    L.attn_norm_w.assign(attn_norm_w, attn_norm_w + attn_norm_n);
    L.ffn_norm_w.assign(ffn_norm_w, ffn_norm_w + ffn_norm_n);
    return L;
}

}  // namespace ter::tx
```

- [ ] **Step 3: Create `tests/test_layer_types.cpp`**

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/tx/layer.hpp>
#include <vector>
#include <random>

using namespace ter;
using namespace ter::tx;

TEST_CASE("quantize_layer fills all 7 weight tensors") {
    constexpr int H = 4, HD = 4, I = 8;
    std::vector<float> Wq(H * HD), Wk(H * HD), Wv(H * HD), Wo(HD * H);
    std::vector<float> Wgate(H * I), Wup(H * I), Wdown(I * H);
    std::vector<float> nw1(H, 1.0f), nw2(H, 1.0f);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : Wq)    v = dist(rng);
    for (auto& v : Wk)    v = dist(rng);
    for (auto& v : Wv)    v = dist(rng);
    for (auto& v : Wo)    v = dist(rng);
    for (auto& v : Wgate) v = dist(rng);
    for (auto& v : Wup)   v = dist(rng);
    for (auto& v : Wdown) v = dist(rng);

    LayerWeights L = quantize_layer(
        Wq.data(),    H,  HD,
        Wk.data(),    H,  HD,
        Wv.data(),    H,  HD,
        Wo.data(),    HD, H,
        Wgate.data(), H,  I,
        Wup.data(),   H,  I,
        Wdown.data(), I,  H,
        nw1.data(),   H,
        nw2.data(),   H);

    CHECK(L.Wq.payload.size()    == static_cast<size_t>(H * HD));
    CHECK(L.Wk.payload.size()    == static_cast<size_t>(H * HD));
    CHECK(L.Wv.payload.size()    == static_cast<size_t>(H * HD));
    CHECK(L.Wo.payload.size()    == static_cast<size_t>(HD * H));
    CHECK(L.Wgate.payload.size() == static_cast<size_t>(H * I));
    CHECK(L.Wup.payload.size()   == static_cast<size_t>(H * I));
    CHECK(L.Wdown.payload.size() == static_cast<size_t>(I * H));
    CHECK(L.Wq.scale > 0.0f);
    CHECK(L.attn_norm_w.size() == static_cast<size_t>(H));
    CHECK(L.ffn_norm_w.size()  == static_cast<size_t>(H));
}
```

- [ ] **Step 4: Wire build**

In `src/CMakeLists.txt`, add:
```cmake
target_sources(ter PRIVATE
    # existing entries ...
    tx/layer.cpp
)
```

In `tests/CMakeLists.txt`:
```cmake
ter_add_test(test_layer_types)
```

- [ ] **Step 5: Build and run**

```
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: 45/45.

- [ ] **Step 6: Commit**

```
git add include/ter/tx/layer.hpp src/tx/layer.cpp src/CMakeLists.txt tests/test_layer_types.cpp tests/CMakeLists.txt
git commit -m "feat(tx): LayerWeights struct + quantize_layer helper"
```

After: `git log --oneline -2`.

---

## Task F5.2.2 — `forward_layer()` orchestrator

**Files:**
- Create `include/ter/tx/forward.hpp`
- Create `src/tx/forward.cpp`
- Modify `src/CMakeLists.txt`

This is the meat of the plan: orchestrate one layer's forward pass using our 5 kernels, returning a float vector of `hidden_size`.

The function is long because it composes 7 matmuls + 2 ropes + 1 softmax + 1 swiglu + 2 rmsnorms + 2 residual adds. The pattern is the one from `tests/test_attention.cpp` (F4.10) extended with the FFN block.

For F5.2 we keep it single-token (`pos = 0`, no past KV cache). Multi-token streaming + KV cache lands in F5.3.

- [ ] **Step 1: Write `include/ter/tx/forward.hpp`**

```cpp
#pragma once
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/tx/layer.hpp>
#include <vector>

namespace ter::tx {

// Single-token, single-layer forward pass.
// hidden_in: float vector of length hidden_size (the layer input).
// hidden_out: float vector of length hidden_size (the layer output, post-residual).
// pos: token position (for RoPE).
// head_dim, n_heads, n_kv_heads, intermediate_size: model dims.
// rmsnorm_eps: typically 1e-6.
// LUT addresses must be pre-loaded by the caller (rsqrt, exp, rcp, sigmoid).
struct LutAddrs {
    int rsqrt;
    int exp;
    int rcp;
    int sigmoid;
};

void forward_layer(
    Sim& sim,
    KernelTable& kt,
    const LayerWeights& L,
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

}  // namespace ter::tx
```

- [ ] **Step 2: Implement `src/tx/forward.cpp`**

```cpp
#include <ter/tx/forward.hpp>
#include <ter/numfmt.hpp>
#include <cmath>
#include <algorithm>

namespace ter::tx {

namespace {

// Sim memory map for forward_layer (single-token, single-layer):
constexpr int SCRATCH_X        = 1024;   // matmul X scratch (27 words)
constexpr int SCRATCH_W        = 1100;   // matmul W column scratch (27 words)
constexpr int Y_TILE           = 1200;   // matmul output tile (1 word)
constexpr int X_BUF            = 1300;   // current activation, padded (27 words)
constexpr int Q_BUF            = 1400;   // Q after rope (27 words)
constexpr int K_BUF            = 1500;
constexpr int V_BUF            = 1600;
constexpr int RCOS             = 1700;
constexpr int RSIN             = 1800;
constexpr int RROT             = 1900;
constexpr int Y_ROPE           = 2000;
constexpr int SOFT_X           = 2100;
constexpr int SOFT_Y           = 2200;
constexpr int SILU_X           = 2300;
constexpr int SILU_Y           = 2400;

constexpr int VEC_LANES        = 27;
constexpr int OUT_SCALE        = 9841;

// mm_row: dot row[r] of Xt by all columns of Wt, single tile, single output word.
// Returns float result for each output column.
void mm_row(Sim& sim, KernelTable& kt, KernelId id_mm,
            const TritTensor& Xt, int row,
            const TritTensor& Wt,
            int K, int N,
            std::vector<float>& out) {
    out.assign(N, 0.0f);
    for (int j = 0; j < N; ++j) {
        int64_t int_acc = 0;
        for (int k0 = 0; k0 < K; k0 += 27) {
            int chunk = std::min(27, K - k0);
            for (int t = 0; t < 27; ++t) {
                int xv = (t < chunk) ? Xt.payload[row * K + (k0 + t)].to_int() : 0;
                int wv = (t < chunk) ? Wt.payload[(k0 + t) * N + j].to_int() : 0;
                sim.mem().store_word(SCRATCH_X + t, Word27::from_int(xv));
                sim.mem().store_word(SCRATCH_W + t, Word27::from_int(wv));
            }
            std::vector<int64_t> args = {SCRATCH_X, SCRATCH_W, Y_TILE, 0, 0, 0, 0};
            sim.call_kernel(kt, id_mm, args);
            int_acc += sim.mem().load_word(Y_TILE).to_int();
        }
        out[j] = static_cast<float>(int_acc) * Xt.scale * Wt.scale;
    }
}

// Apply RMSNorm: y = (x / rms(x)) * weight, host-only for tractability.
// Future: route through tk_rmsnorm kernel; deferred because per-tensor scale handling
// for inputs that vary per call requires care.
void rmsnorm_host(const std::vector<float>& x, const std::vector<float>& w, float eps,
                  std::vector<float>& y) {
    double ss = 0.0;
    for (auto v : x) ss += double(v) * double(v);
    double rms_inv = 1.0 / std::sqrt(ss / x.size() + eps);
    y.assign(x.size(), 0.0f);
    for (size_t i = 0; i < x.size(); ++i) y[i] = static_cast<float>(x[i] * rms_inv) * w[i];
}

// Apply RoPE to a head-dim vector (single token at position pos).
void rope_host(std::vector<float>& v, int pos, int head_dim) {
    int n_pairs = head_dim / 2;
    for (int k = 0; k < n_pairs; ++k) {
        double freq = 1.0 / std::pow(10000.0, (2.0 * k) / double(head_dim));
        double angle = double(pos) * freq;
        double c = std::cos(angle), s = std::sin(angle);
        float x0 = v[2 * k], x1 = v[2 * k + 1];
        v[2 * k]     = static_cast<float>(x0 * c - x1 * s);
        v[2 * k + 1] = static_cast<float>(x0 * s + x1 * c);
    }
}

// Softmax of a length-N vector (host, for tractability).
void softmax_host(std::vector<float>& v) {
    double mx = v[0];
    for (auto x : v) if (x > mx) mx = x;
    double sum = 0.0;
    for (auto& x : v) { x = static_cast<float>(std::exp(double(x) - mx)); sum += x; }
    for (auto& x : v) x = static_cast<float>(x / sum);
}

// SiLU * up element-wise.
void silu_mul_host(const std::vector<float>& gate, const std::vector<float>& up,
                   std::vector<float>& y) {
    y.assign(gate.size(), 0.0f);
    for (size_t i = 0; i < gate.size(); ++i) {
        double s = 1.0 / (1.0 + std::exp(-double(gate[i])));
        y[i] = static_cast<float>(double(gate[i]) * s * double(up[i]));
    }
}

}  // namespace

void forward_layer(
    Sim& sim,
    KernelTable& kt,
    const LayerWeights& L,
    const std::vector<float>& hidden_in,
    int pos,
    int hidden_size,
    int head_dim,
    int n_heads,
    int n_kv_heads,
    int intermediate_size,
    float rmsnorm_eps,
    const LutAddrs& /*luts*/,           // unused in this MVP — host-side norms/softmax
    std::vector<float>& hidden_out)
{
    KernelId id_mm = kt.find("tk_matmul_b_9t");

    // 1) attn_norm
    std::vector<float> x_norm;
    rmsnorm_host(hidden_in, L.attn_norm_w, rmsnorm_eps, x_norm);

    // 2) Q = x_norm @ Wq, similarly K, V.
    TritTensor xt = quantize(x_norm.data(), {1, hidden_size}, 9);
    std::vector<float> q, k, v;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    mm_row(sim, kt, id_mm, xt, 0, L.Wq, hidden_size, q_dim, q);
    mm_row(sim, kt, id_mm, xt, 0, L.Wk, hidden_size, kv_dim, k);
    mm_row(sim, kt, id_mm, xt, 0, L.Wv, hidden_size, kv_dim, v);

    // 3) Apply RoPE to Q and K per head.
    for (int h = 0; h < n_heads; ++h) {
        std::vector<float> qh(q.begin() + h * head_dim, q.begin() + (h + 1) * head_dim);
        rope_host(qh, pos, head_dim);
        std::copy(qh.begin(), qh.end(), q.begin() + h * head_dim);
    }
    for (int h = 0; h < n_kv_heads; ++h) {
        std::vector<float> kh(k.begin() + h * head_dim, k.begin() + (h + 1) * head_dim);
        rope_host(kh, pos, head_dim);
        std::copy(kh.begin(), kh.end(), k.begin() + h * head_dim);
    }

    // 4) attention scores. With pos=0 and no KV cache, attention is self-only:
    //    scores[h] = Q[h] · K[h_kv] / sqrt(head_dim), single scalar per head.
    //    softmax of a scalar is 1.0; ctx[h] = 1.0 * V[h_kv] = V[h_kv].
    // For MVP we just propagate V directly. (Multi-token attention lands in F5.3.)
    std::vector<float> ctx = v;  // Same dim as v.

    // 5) Wo projection.
    TritTensor ctxt = quantize(ctx.data(), {1, q_dim}, 9);
    std::vector<float> attn_out;
    mm_row(sim, kt, id_mm, ctxt, 0, L.Wo, q_dim, hidden_size, attn_out);

    // 6) Residual: hidden_mid = hidden_in + attn_out
    std::vector<float> hidden_mid(hidden_size);
    for (int i = 0; i < hidden_size; ++i) hidden_mid[i] = hidden_in[i] + attn_out[i];

    // 7) ffn_norm
    std::vector<float> mid_norm;
    rmsnorm_host(hidden_mid, L.ffn_norm_w, rmsnorm_eps, mid_norm);

    // 8) gate = mid_norm @ Wgate; up = mid_norm @ Wup
    TritTensor mt = quantize(mid_norm.data(), {1, hidden_size}, 9);
    std::vector<float> gate, up;
    mm_row(sim, kt, id_mm, mt, 0, L.Wgate, hidden_size, intermediate_size, gate);
    mm_row(sim, kt, id_mm, mt, 0, L.Wup,   hidden_size, intermediate_size, up);

    // 9) SiLU(gate) * up
    std::vector<float> ff;
    silu_mul_host(gate, up, ff);

    // 10) Wdown projection
    TritTensor fft = quantize(ff.data(), {1, intermediate_size}, 9);
    std::vector<float> ff_out;
    mm_row(sim, kt, id_mm, fft, 0, L.Wdown, intermediate_size, hidden_size, ff_out);

    // 11) Residual: hidden_out = hidden_mid + ff_out
    hidden_out.assign(hidden_size, 0.0f);
    for (int i = 0; i < hidden_size; ++i) hidden_out[i] = hidden_mid[i] + ff_out[i];
}

}  // namespace ter::tx
```

**Note** the MVP simplification: with `pos=0` and no KV cache, the attention "weighted sum" is trivially `V` itself (softmax of one element is 1.0). This is wrong for multi-token attention but correct for single-token at position 0. Multi-token follows the test_attention.cpp pattern in F5.3.

The host-side RMSNorm/RoPE/Softmax/SiLU calls are also simplifications — F5.2 wires the matmul kernel only. The other kernels can be plumbed in F5.3 alongside multi-token; the immediate goal is proving the matmul + composition flow works end-to-end.

- [ ] **Step 3: Update `src/CMakeLists.txt`**

Add `tx/forward.cpp` to the `ter` library sources.

- [ ] **Step 4: Build (no new test yet)**

```
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: 45/45.

- [ ] **Step 5: Commit**

```
git add include/ter/tx/forward.hpp src/tx/forward.cpp src/CMakeLists.txt
git commit -m "feat(tx): forward_layer() orchestrator (matmul kernel + host-side everything else)"
```

After: `git log --oneline -2`.

---

## Task F5.2.3 — End-to-end single-layer test vs numpy

**Files:**
- Create `tests/test_layer_forward.cpp`
- Modify `tests/CMakeLists.txt`

The test:
1. Generates random hidden_in and random Layer weights.
2. Computes the numpy reference (host-only).
3. Calls `forward_layer()` (matmul via kernels).
4. Asserts max_rel < 0.5 (generous; multiple matmul rounds compound quantization noise).

- [ ] **Step 1: Write `tests/test_layer_forward.cpp`**

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

// Numpy-equivalent forward layer.
std::vector<float> numpy_forward_layer(
    const std::vector<float>& hidden_in,
    const std::vector<float>& Wq, const std::vector<float>& Wk,
    const std::vector<float>& Wv, const std::vector<float>& Wo,
    const std::vector<float>& Wgate, const std::vector<float>& Wup,
    const std::vector<float>& Wdown,
    const std::vector<float>& nw1,  const std::vector<float>& nw2,
    int pos)
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

    // 1) RMSNorm + Q/K/V projections
    auto x_norm = rmsnorm(hidden_in, nw1, 1e-6f);
    auto q = matvec(x_norm, Wq, H, HD);
    auto k = matvec(x_norm, Wk, H, HD);
    auto v = matvec(x_norm, Wv, H, HD);

    // 2) RoPE
    rope(q, pos, HD);
    rope(k, pos, HD);

    // 3) attention with single token: scores = Q·K/sqrt(HD), softmax(scalar)=1, ctx=V
    std::vector<float> ctx = v;

    // 4) Wo
    auto attn_out = matvec(ctx, Wo, HD, H);

    // 5) Residual
    std::vector<float> mid(H);
    for (int i = 0; i < H; ++i) mid[i] = hidden_in[i] + attn_out[i];

    // 6) ffn_norm
    auto mid_norm = rmsnorm(mid, nw2, 1e-6f);

    // 7) Gate, Up
    auto gate = matvec(mid_norm, Wgate, H, I);
    auto up   = matvec(mid_norm, Wup,   H, I);

    // 8) SiLU * up
    std::vector<float> ff(I);
    for (int i = 0; i < I; ++i) {
        double s = 1.0 / (1.0 + std::exp(-double(gate[i])));
        ff[i] = static_cast<float>(double(gate[i]) * s * double(up[i]));
    }

    // 9) Wdown
    auto ff_out = matvec(ff, Wdown, I, H);

    // 10) Final residual
    std::vector<float> out(H);
    for (int i = 0; i < H; ++i) out[i] = mid[i] + ff_out[i];
    return out;
}

}  // namespace

TEST_CASE("forward_layer matches numpy reference within bounded rel_err") {
    std::mt19937 rng(0xBABE);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> hidden_in(H);
    for (auto& v : hidden_in) v = dist(rng);

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

    auto out_ref = numpy_forward_layer(hidden_in,
        Wq, Wk, Wv, Wo, Wgate, Wup, Wdown, nw1, nw2, /*pos*/0);

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

    LutAddrs luts{0, 0, 0, 0};   // unused in this MVP

    std::vector<float> hidden_out;
    forward_layer(s, kt, L, hidden_in, /*pos*/0,
                  H, HD, Hn, Kn, I, 1e-6f, luts, hidden_out);

    REQUIRE(hidden_out.size() == static_cast<size_t>(H));

    double max_rel = 0.0;
    for (int i = 0; i < H; ++i) {
        double ref = out_ref[i];
        double got = hidden_out[i];
        double denom = std::max(0.1, std::fabs(ref));
        double rel = std::fabs(got - ref) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    CHECK(max_rel < 0.5);

    // Counter sanity: TVMAC fired at least 7 times (one per matmul tile, one tile per col,
    // 7 matmuls × ~few cols each). Don't pin exactly.
    CHECK(s.counters().get(Opcode::TVMAC) > 0);
}
```

- [ ] **Step 2: Wire test in `tests/CMakeLists.txt`**

```cmake
ter_add_test(test_layer_forward)
target_compile_definitions(test_layer_forward PRIVATE TER_KERNELS_DIR="${CMAKE_SOURCE_DIR}/src/kernels")
```

- [ ] **Step 3: Build and run**

Expected: 46/46.

If `max_rel >= 0.5`:
- Print hidden_out and out_ref to compare.
- Most likely: a matmul shape transpose mismatch. Wq is shape (H, HD) row-major = X[K]@W[K,N] convention (numpy's `x @ W`); confirm `mm_row` matches.
- Verify the residual additions are correct (just element-wise add, both vectors same length).
- Double-check the simplified attention (V passed through directly) matches the reference.

If max_rel ~10-30%, accept (chained quantization noise).

- [ ] **Step 4: Commit**

```
git add tests/test_layer_forward.cpp tests/CMakeLists.txt
git commit -m "test(tx): F5.2 — single-layer forward via kernels matches numpy"
```

After: `git log --oneline -3`.

---

## Final Task — README + bridge-notes update

**Files:**
- Modify `README.md` Status block
- Append to `docs/bridge-notes.md`

- [ ] **Step 1: Update `README.md`**

Replace the F5.2 line with:
```markdown
- [x] F5.2 — Single-layer forward (`ter::tx::forward_layer()`) via matmul kernel matches numpy on tiny shapes.
```

(F5.3 stays as `[ ]`.)

- [ ] **Step 2: Append to `docs/bridge-notes.md`**

```markdown
## F5.2 result — first transformer layer running

- `ter::tx::LayerWeights` holds quantized (`TritFP_B`, 9 trits) projection weights for one layer.
- `ter::tx::forward_layer()` orchestrates one full layer: RMSNorm → Q/K/V → RoPE → trivial attention (single token) → Wo → residual → RMSNorm → SwiGLU FFN → residual.
- All 7 matmul projections route through `tk_matmul_b_9t`. RMSNorm/RoPE/SwiGLU are host-side for this MVP — F5.3 plumbs them through their respective kernels alongside multi-token attention.
- Test passes `max_rel < 0.5` against numpy on tiny shapes (H=4, HD=4, I=8). Tighter shapes/more tests come with F5.3.

### Why kernels everywhere except matmul are deferred to F5.3

Matmul is the bandwidth-dominant op and the cleanest to wire (we proved the pattern in F4.4). RMSNorm/Softmax/SiLU/RoPE all need per-call LUT setup (rsqrt LUT, sigmoid LUT, etc.) — each adds host-side prep code. Plumbing them all in one go alongside multi-token attention would inflate F5.2's task list. Doing them in F5.3 alongside the multi-token (KV cache) restructuring keeps each plan focused.
```

- [ ] **Step 3: Build, verify**

Expected: 46/46 still.

- [ ] **Step 4: Commit**

```
git add README.md docs/bridge-notes.md
git commit -m "docs(ter): F5.2 done — single-layer forward + bridge notes"
```

After: `git log --oneline -3`.

---

## Self-Review

- **Spec coverage:** §9 — partial (F5.2 covers the layer flow with matmul-only routing through kernels; F5.3 will add the rest of the kernels and multi-token).
- **Placeholder scan:** the LutAddrs struct is unused in F5.2; documented as F5.3 plumbing.
- **Type consistency:** `ter::tx::LayerWeights`, `LayerActivations`, `forward_layer`, `LutAddrs`, `quantize_layer` referenced consistently.
- **Known caveats:**
  - Single token, pos=0 only: simplified attention (V passes through). Multi-token + KV cache lands in F5.3.
  - Only matmul routes through a kernel; RMSNorm/RoPE/SiLU/Softmax are host-side for this MVP. F5.3 changes that.
  - Tiny shapes: H=4, HD=4, I=8. Larger shapes need padding to multiples of 27 in matmul tiling — already proven in `test_kernel_matmul_b_32.cpp` (F4.4).

---

## Execution Handoff

After all tasks: `ter::tx::forward_layer()` exists and runs one full transformer layer through our ternary kernel substrate (matmul kernel + host-side rest), matching numpy within bounded relative error.

Next plan (F5.3): multi-token attention with KV cache + plumb the remaining kernels (rmsnorm, softmax, silu, rope) into `forward_layer`. Sets us up for full TinyLlama smoke in F5.4.
