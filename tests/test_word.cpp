#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/word.hpp>
#include <cstdlib>

using namespace ter;

TEST_CASE("Word27 round-trip from int64") {
    int64_t cases[] = {0, 1, -1, 42, -42, 7625597484987LL, -7625597484987LL};
    for (int64_t v : cases) {
        if (std::abs(v) <= Word27::max_int()) {
            CHECK(Word27::from_int(v).to_int() == v);
        }
    }
}

TEST_CASE("Word27 add and sub") {
    auto a = Word27::from_int(1234567);
    auto b = Word27::from_int(-89012);
    auto c = Word27::from_int(345);
    CHECK((a + b).to_int() == 1234567 - 89012);
    CHECK((a - b).to_int() == 1234567 + 89012);
    CHECK((((a + b) + c)) == ((a + (b + c))));
}

TEST_CASE("Word27 negation") {
    auto x = Word27::from_int(987654);
    CHECK((-x).to_int() == -987654);
    CHECK((-(-x)) == x);
}

TEST_CASE("Word27 sign_trit") {
    CHECK(sign_trit(Word27::from_int(100)) == T_POS);
    CHECK(sign_trit(Word27::from_int(-100)) == T_NEG);
    CHECK(sign_trit(Word27::from_int(0)) == T_ZERO);
}

TEST_CASE("Word27 (a + b) + c == a + (b + c) -- sampled") {
    int64_t samples[] = {0, 1, -1, 100, -100, 1234567, -1234567, 99999999};
    for (auto a : samples) for (auto b : samples) for (auto c : samples) {
        auto wa = Word27::from_int(a);
        auto wb = Word27::from_int(b);
        auto wc = Word27::from_int(c);
        if (std::abs(a + b + c) <= Word27::max_int()) {
            CHECK(((wa + wb) + wc) == (wa + (wb + wc)));
        }
    }
}

TEST_CASE("Word27 -(-x) == x") {
    for (int64_t v : {0LL, 1LL, -1LL, 99999999LL, -99999999LL}) {
        auto w = Word27::from_int(v);
        CHECK((-(-w)) == w);
    }
}
