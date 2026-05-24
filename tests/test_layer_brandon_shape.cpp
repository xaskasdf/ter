// Verifies forward_layer's host-side fallback paths work at real brandon-tiny shapes
// (H=256, HD=32, I=720). Uses random weights so we don't need the GGUF — just exercises
// the hidden_size>27 / intermediate_size>27 / head_dim>26 dispatch branches.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/tx/forward.hpp>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/tx/lut_setup.hpp>
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
constexpr int H  = 256;     // brandon dim
constexpr int HD = 32;      // brandon head_dim (>26 → forces host rope)
constexpr int Hn = 8;       // brandon n_heads
constexpr int Kn = 2;       // brandon n_kv_heads (GQA group=4)
constexpr int I  = 720;     // brandon intermediate (>27 → forces host silu)

// Small max_seq for the test; cache only needs to hold a few positions.
constexpr int MAX_SEQ = 4;
}  // namespace

TEST_CASE("forward_layer at brandon shapes (host fallbacks engaged)") {
    std::mt19937 rng(0xBEEFu);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);   // small to keep matmul output bounded

    std::vector<float> hidden_in(H);
    for (auto& v : hidden_in) v = dist(rng);

    std::vector<float> Wq(H * Hn * HD), Wk(H * Kn * HD), Wv(H * Kn * HD);
    std::vector<float> Wo(Hn * HD * H);
    std::vector<float> Wgate(H * I), Wup(H * I), Wdown(I * H);
    std::vector<float> nw1(H, 1.0f), nw2(H, 1.0f);
    auto fill = [&](std::vector<float>& w) { for (auto& v : w) v = dist(rng); };
    fill(Wq); fill(Wk); fill(Wv); fill(Wo);
    fill(Wgate); fill(Wup); fill(Wdown);

    LayerWeights L = quantize_layer(
        Wq.data(),    H, Hn * HD,
        Wk.data(),    H, Kn * HD,
        Wv.data(),    H, Kn * HD,
        Wo.data(),    Hn * HD, H,
        Wgate.data(), H, I,
        Wup.data(),   H, I,
        Wdown.data(), I, H,
        nw1.data(),   H,
        nw2.data(),   H);

    // Sim memory budget: scratch addrs go up to ~2200; LUTs at 5000-6900;
    // KV cache lives in host vectors not sim memory. 16K words is plenty.
    Sim s(16 * 1024);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);

    LutAddrs luts = load_default_luts(s, "lut_data");

    KVCache cache;
    cache.resize(MAX_SEQ, Kn, HD);

    std::vector<float> hidden_out;
    forward_layer(s, kt, L, cache, hidden_in, /*pos=*/0,
                  H, HD, Hn, Kn, I, /*eps=*/1e-6f, /*rope_theta=*/10000.0, /*ffn_relu2=*/false, luts, hidden_out);

    REQUIRE(hidden_out.size() == static_cast<size_t>(H));

    // Sanity: output must be finite (no NaN/Inf from the host paths).
    int n_finite = 0;
    for (float v : hidden_out) if (std::isfinite(v)) ++n_finite;
    CHECK(n_finite == H);

    // Sanity: output must not be identically zero (means matmul/host path broke).
    double abs_sum = 0.0;
    for (float v : hidden_out) abs_sum += std::fabs(static_cast<double>(v));
    CHECK(abs_sum > 0.0);

    MESSAGE("brandon-shape forward_layer: |out|_1 = ", abs_sum);

    // Counter sanity: at brandon shapes, lots of matmul tiles fired.
    // mm_row over (1xH) @ (HxN) is ceil(H/27) tiles per output column;
    // total tvmacs >= sum over the 7 projections. For H=256, Hn*HD=256, etc:
    //   Wq:    256/27 * 256 ≈ 2560 tiles
    //   Wk:    256/27 *  64 ≈  640
    //   Wv:    256/27 *  64 ≈  640
    //   Wo:    256/27 * 256 ≈ 2560
    //   Wgate: 256/27 * 720 ≈ 7200
    //   Wup:   256/27 * 720 ≈ 7200
    //   Wdown: 720/27 * 256 ≈ 7168
    // Total ≈ 27 968 tvmacs. Pin a generous lower bound.
    auto tvmac = s.counters().get(Opcode::TVMAC);
    MESSAGE("TVMAC count = ", tvmac);
    CHECK(tvmac > 20000);
}
