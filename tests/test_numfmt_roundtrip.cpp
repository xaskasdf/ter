#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/numfmt.hpp>
#include <random>
#include <cmath>
#include <vector>

using namespace ter;

static double mse(const std::vector<float>& a, const std::vector<float>& b) {
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = double(a[i]) - double(b[i]);
        acc += d * d;
    }
    return acc / static_cast<double>(a.size());
}

TEST_CASE("round-trip MSE under quantization-noise bound (uniform [-1,1])") {
    constexpr int N = 4096;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> xs(N);
    for (auto& v : xs) v = u(rng);

    TritTensor t = quantize(xs.data(), {N}, 9);
    std::vector<float> ys(N);
    dequantize(t, ys.data());

    // Variance per element <= scale^2/12 (uniform error in [-scale/2, scale/2]).
    // Multiply by 4 to leave headroom; the test catches gross errors.
    double bound = static_cast<double>(t.scale) * static_cast<double>(t.scale) / 12.0 * 4.0;
    CHECK(mse(xs, ys) < bound);
}

TEST_CASE("round-trip MSE drops as n_trits increases") {
    constexpr int N = 1024;
    std::mt19937 rng(67890);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> xs(N);
    for (auto& v : xs) v = u(rng);

    std::vector<float> ys(N);
    TritTensor t9  = quantize(xs.data(), {N}, 9);
    dequantize(t9, ys.data());
    double m9 = mse(xs, ys);

    TritTensor t12 = quantize(xs.data(), {N}, 12);
    dequantize(t12, ys.data());
    double m12 = mse(xs, ys);

    // Each extra trit ~3× resolution → MSE drops ~9×. Allow generous slack.
    CHECK(m12 < m9 / 4.0);
}
