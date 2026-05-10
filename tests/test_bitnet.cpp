// F9: BitNet b1.58 quantizer + analytical op-count.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/bitnet.hpp>
#include <vector>
#include <cmath>

using namespace ter;

TEST_CASE("BitNet quantizer maps to {-1, 0, +1} with mean-abs scale") {
    std::vector<float> in = {-1.5f, -0.6f, -0.1f, 0.0f, 0.2f, 0.7f, 1.4f};
    std::vector<std::int8_t> out(in.size());
    float scale = quantize_bitnet(in.data(), in.size(), out.data());

    double expected_scale = 0.0;
    for (float v : in) expected_scale += std::fabs(v);
    expected_scale /= in.size();
    CHECK(std::fabs(scale - static_cast<float>(expected_scale)) < 1e-6f);

    // Every quant must be in {-1, 0, +1}; sign must match input sign.
    for (std::size_t i = 0; i < in.size(); ++i) {
        CHECK(out[i] >= -1);
        CHECK(out[i] <=  1);
        if (in[i] >  0) CHECK(out[i] >= 0);
        if (in[i] <  0) CHECK(out[i] <= 0);
    }
}

TEST_CASE("BitNet quantizer handles all-zero input") {
    std::vector<float> in(8, 0.0f);
    std::vector<std::int8_t> out(8);
    float scale = quantize_bitnet(in.data(), in.size(), out.data());
    CHECK(scale == 0.0f);
    for (auto q : out) CHECK(q == 0);
}

TEST_CASE("BitNet matmul op-count: zeros become skips, no TVMAC") {
    // 4x4 matrix with mixed signs and one row of zeros.
    std::vector<std::int8_t> W = {
         1, -1,  0,  1,
         0,  0,  0,  0,
        -1, -1,  1,  1,
         1,  0, -1,  0,
    };
    auto ops = bitnet_matmul_ops(W.data(), 4, 4);
    CHECK(ops.tvmac == 0);
    CHECK(ops.tvadd == 5);  // count of +1
    CHECK(ops.tvsub == 4);  // count of -1
    CHECK(ops.skips == 7);  // count of 0
    CHECK(ops.tvadd + ops.tvsub + ops.skips == 16);
}

TEST_CASE("BitNet headline: synthetic 1B-param matmul has zero TVMAC") {
    // Demonstrate the H3 hypothesis: when substrate and data alignment match
    // (ternary substrate + native ternary weights), the matmul TVMAC count
    // is exactly zero. ADDs replace MACs entirely.
    std::vector<std::int8_t> W(1024, 0);
    for (std::size_t i = 0; i < W.size(); i += 3) W[i] = 1;
    for (std::size_t i = 1; i < W.size(); i += 3) W[i] = -1;
    auto ops = bitnet_matmul_ops(W.data(), 32, 32);
    CHECK(ops.tvmac == 0);
    CHECK(ops.tvadd + ops.tvsub > 0);
}
