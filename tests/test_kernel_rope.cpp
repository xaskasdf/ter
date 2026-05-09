#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/numfmt.hpp>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

#ifndef TER_KERNELS_DIR
#error "TER_KERNELS_DIR must be defined"
#endif

using namespace ter;

// RoPE kernel: now that TVADD/TVSUB/TVNEG use wide (int32) clamp (matching
// TVMUL), x can be quantized with the full 9 trits and OUT_SCALE = 9841.
// The TVADD result x*cos_int + rotated_x*sin_int fits int32 comfortably.
// Reference is computed from the original float x (not the quantized version)
// so that the test exercises realistic 9-trit quantization noise.

TEST_CASE("tk_rope rotates a 26-element pair-vector matching numpy reference") {
    constexpr int HEAD_DIM = 26;
    constexpr int N_PAIRS  = HEAD_DIM / 2;
    constexpr int VEC_LANES = 27;
    constexpr int OUT_SCALE = 9841;
    constexpr int POS = 5;

    std::vector<float> x(VEC_LANES, 0.0f);
    std::mt19937 rng(0x1337);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int i = 0; i < HEAD_DIM; ++i) x[i] = dist(rng);

    // Reference: numpy-style RoPE from the original float x.
    std::vector<float> y_ref(VEC_LANES, 0.0f);
    for (int k = 0; k < N_PAIRS; ++k) {
        double freq  = 1.0 / std::pow(10000.0, (2.0 * k) / double(HEAD_DIM));
        double angle = double(POS) * freq;
        double c = std::cos(angle);
        double s = std::sin(angle);
        double x0 = x[2 * k], x1 = x[2 * k + 1];
        y_ref[2 * k]     = static_cast<float>(x0 * c - x1 * s);
        y_ref[2 * k + 1] = static_cast<float>(x0 * s + x1 * c);
    }

    // 9-trit quantize: realistic per-tensor fixed-point encoding.
    TritTensor xt = quantize(x.data(), {VEC_LANES}, 9);

    std::vector<int> cos_vec(VEC_LANES, 0);
    std::vector<int> sin_vec(VEC_LANES, 0);
    std::vector<int> rotated_x(VEC_LANES, 0);
    for (int k = 0; k < N_PAIRS; ++k) {
        double freq  = 1.0 / std::pow(10000.0, (2.0 * k) / double(HEAD_DIM));
        double angle = double(POS) * freq;
        int c_int = static_cast<int>(std::round(std::cos(angle) * OUT_SCALE));
        int s_int = static_cast<int>(std::round(std::sin(angle) * OUT_SCALE));
        cos_vec[2 * k]     = c_int;
        cos_vec[2 * k + 1] = c_int;
        sin_vec[2 * k]     = s_int;
        sin_vec[2 * k + 1] = s_int;

        int x0_int = static_cast<int>(xt.payload[2 * k]);
        int x1_int = static_cast<int>(xt.payload[2 * k + 1]);
        rotated_x[2 * k]     = -x1_int;
        rotated_x[2 * k + 1] = x0_int;
    }

    Sim s(8192);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    KernelId id = kt.find("tk_rope");
    REQUIRE(id.valid);

    int x_addr = 1000, cos_addr = 1100, sin_addr = 1200, rotx_addr = 1300, y_addr = 1400;
    for (int i = 0; i < VEC_LANES; ++i) {
        s.mem().store_word(static_cast<size_t>(x_addr    + i), ter::Word27::from_int(xt.payload[i]));
        s.mem().store_word(static_cast<size_t>(cos_addr  + i), Word27::from_int(cos_vec[i]));
        s.mem().store_word(static_cast<size_t>(sin_addr  + i), Word27::from_int(sin_vec[i]));
        s.mem().store_word(static_cast<size_t>(rotx_addr + i), Word27::from_int(rotated_x[i]));
    }

    std::vector<int64_t> args = {x_addr, cos_addr, sin_addr, rotx_addr, y_addr, 0, 0};
    s.call_kernel(kt, id, args);

    // recovery = xt.scale / OUT_SCALE converts integer lane values back to float.
    float recovery = xt.scale / static_cast<float>(OUT_SCALE);
    std::vector<float> y(VEC_LANES);
    for (int i = 0; i < VEC_LANES; ++i) {
        y[i] = static_cast<float>(
                   s.mem().load_word(static_cast<size_t>(y_addr + i)).to_int())
               * recovery;
    }

    double max_rel = 0.0;
    for (int i = 0; i < HEAD_DIM; ++i) {
        double ref   = y_ref[i];
        double got   = y[i];
        double denom = std::max(1e-3, std::fabs(ref));
        double rel   = std::fabs(got - ref) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    CHECK(max_rel < 1e-1);  // generous: full 9-trit quantization noise included

    CHECK(s.counters().get(Opcode::TVLOAD)  == 4);
    CHECK(s.counters().get(Opcode::TVMUL)   == 2);
    CHECK(s.counters().get(Opcode::TVADD)   == 1);
    CHECK(s.counters().get(Opcode::TVSTORE) == 1);
}
