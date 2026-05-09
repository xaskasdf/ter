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
    for (std::size_t i = 0; i < in.size(); ++i) CHECK(out[i] == in[i]);
}
