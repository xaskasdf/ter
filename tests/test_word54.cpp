#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/word.hpp>

using namespace ter;

TEST_CASE("Word54 round-trip") {
    int64_t cases[] = {0, 1, -1, 1000000, -1000000, 9999999, -9999999};
    for (int64_t v : cases) {
        Word54 w = Word54::from_int(v);
        CHECK(w.to_int() == v);
    }
}

TEST_CASE("Word54 = Word27 * Word27") {
    auto a = Word27::from_int(123456);
    auto b = Word27::from_int(-789);
    Word54 prod = mul(a, b);
    CHECK(prod.to_int() == int64_t{123456} * int64_t{-789});
}

TEST_CASE("Word54 mac_inplace") {
    Word54 acc = Word54::from_int(1000);
    auto a = Word27::from_int(7);
    auto b = Word27::from_int(11);
    mac_inplace(acc, a, b);
    CHECK(acc.to_int() == 1000 + 77);
}
