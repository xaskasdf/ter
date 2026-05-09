#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/tryte.hpp>

using namespace ter;

TEST_CASE("Tryte round-trip int <-> tryte") {
    for (int v = -13; v <= 13; ++v) {
        Tryte t = Tryte::from_int(v);
        CHECK(t.to_int() == v);
    }
}

TEST_CASE("Tryte range") {
    CHECK(Tryte::min_int() == -13);
    CHECK(Tryte::max_int() == +13);
}

TEST_CASE("Tryte negation") {
    for (int v = -13; v <= 13; ++v) {
        Tryte t = Tryte::from_int(v);
        CHECK((-t).to_int() == -v);
    }
}
