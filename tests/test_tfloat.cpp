// F7: Format A (tfloat) round-trip and band-positioning tests.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/tfloat.hpp>
#include <cmath>

using ter::TFloat;

TEST_CASE("tfloat: zero round-trips exactly") {
    auto t = TFloat::from_float(0.0f);
    CHECK(t.is_zero());
    CHECK(t.to_float() == 0.0f);
}

TEST_CASE("tfloat: small integers round-trip near-exactly") {
    for (float x : {1.0f, -1.0f, 3.0f, -3.0f, 9.0f, 27.0f, 81.0f}) {
        auto t = TFloat::from_float(x);
        float r = t.to_float();
        // Powers of 3 should be exact; integers in between within mantissa precision.
        CHECK(std::fabs(r - x) <= std::fabs(x) * 1e-3f + 1e-6f);
    }
}

TEST_CASE("tfloat: encode places mantissa in [-MANT_MAX, MANT_MAX]") {
    for (float x : {1e-5f, 1e-2f, 1.0f, 1e3f, 1e5f, -3.14f, -1e10f}) {
        auto t = TFloat::from_float(x);
        CHECK(t.mantissa >= -TFloat::MANT_MAX);
        CHECK(t.mantissa <=  TFloat::MANT_MAX);
        CHECK(t.exp >= -TFloat::EXP_MAX);
        CHECK(t.exp <=  TFloat::EXP_MAX);
    }
}

TEST_CASE("tfloat: relative error under 1/MANT_MAX over a wide range") {
    // 9-trit mantissa => relative precision ~ 1/9841 ~ 1e-4.
    constexpr float TOL = 2.0f / static_cast<float>(TFloat::MANT_MAX);  // ~2e-4 with rounding
    for (float x : {0.001f, 0.123f, 1.5f, 17.4f, 314.0f, 1e4f, -1e-3f, -42.42f}) {
        auto t = TFloat::from_float(x);
        float r = t.to_float();
        float rel = std::fabs((r - x) / x);
        CHECK(rel < TOL);
    }
}

TEST_CASE("tfloat: dynamic range comfortably exceeds float32") {
    // tfloat dynamic range: ~5e-58 to ~5e61 (much wider than float32's ~1e-38..3e38).
    // No float input should ever clamp; verify the largest/smallest finite floats
    // stay well within tfloat's exponent range.
    auto big   = TFloat::from_float(3.0e38f);   // near float32 max
    auto small = TFloat::from_float(1.5e-38f);  // near float32 min normal
    CHECK(std::abs(big.exp)   < TFloat::EXP_MAX);
    CHECK(std::abs(small.exp) < TFloat::EXP_MAX);
    CHECK(std::fabs(big.to_float()   / 3.0e38f  - 1.0f) < 2e-4f);
    CHECK(std::fabs(small.to_float() / 1.5e-38f - 1.0f) < 2e-4f);
}
