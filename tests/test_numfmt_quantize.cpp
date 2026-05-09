#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/numfmt.hpp>
#include <vector>
#include <cmath>

using namespace ter;

TEST_CASE("quantize a small float vector with positive max") {
    std::vector<float> xs = {0.1f, -0.5f, 0.25f, 1.0f};
    TritTensor t = quantize(xs.data(), {4}, 9);
    CHECK(t.shape == std::vector<int>{4});
    CHECK(t.n_trits_per_elem == 9);
    CHECK(t.scale == doctest::Approx(1.0f / 9841.0f));
    REQUIRE(t.payload.size() == 4);
    int64_t last = t.payload[3];
    CHECK(last >= 9000);
    CHECK(last <= 9841);
}

TEST_CASE("quantize a vector of zeros yields zero scale") {
    std::vector<float> xs = {0.0f, 0.0f, 0.0f};
    TritTensor t = quantize(xs.data(), {3}, 9);
    CHECK(t.scale == 0.0f);
    for (auto w : t.payload) CHECK(w == 0);
}

TEST_CASE("quantize handles negative-only input") {
    std::vector<float> xs = {-0.5f, -1.0f, -0.25f};
    TritTensor t = quantize(xs.data(), {3}, 9);
    CHECK(t.scale > 0.0f);
    CHECK(t.payload[0] < 0);
    CHECK(t.payload[1] < 0);
}

TEST_CASE("dequantize is approximate inverse of quantize") {
    std::vector<float> xs = {0.1f, -0.5f, 0.25f, 1.0f, -0.75f};
    TritTensor t = quantize(xs.data(), {5}, 9);
    std::vector<float> ys(5);
    dequantize(t, ys.data());
    for (size_t i = 0; i < xs.size(); ++i) {
        CHECK(std::fabs(ys[i] - xs[i]) <= t.scale * 0.5f + 1e-6f);
    }
}

TEST_CASE("dequantize zero tensor returns zeros") {
    std::vector<float> xs = {0.0f, 0.0f};
    TritTensor t = quantize(xs.data(), {2}, 9);
    std::vector<float> ys(2);
    dequantize(t, ys.data());
    CHECK(ys[0] == 0.0f);
    CHECK(ys[1] == 0.0f);
}
