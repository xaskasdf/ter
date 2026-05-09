#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/numfmt.hpp>

using namespace ter;

TEST_CASE("DType enum values are stable") {
    CHECK(static_cast<int>(DType::Float32) == 0);
    CHECK(static_cast<int>(DType::TritFP_B) == 1);
}

TEST_CASE("TritTensor default-constructible and inspectable") {
    TritTensor t;
    CHECK(t.dtype == DType::TritFP_B);
    CHECK(t.n_trits_per_elem == 9);
    CHECK(t.scale == 0.0f);
    CHECK(t.shape.empty());
    CHECK(t.payload.empty());
}

TEST_CASE("TritTensor can be sized and shaped") {
    TritTensor t;
    t.shape = {4, 8};
    t.n_trits_per_elem = 9;
    t.payload.resize(4 * 8);
    CHECK(t.num_elems() == 32);
    CHECK(t.payload.size() == 32);
}
