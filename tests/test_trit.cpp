#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/trit.hpp>

using namespace ter;

TEST_CASE("Trit construction clamps to {-1, 0, +1}") {
    CHECK(Trit{-5}.value() == -1);
    CHECK(Trit{0}.value() == 0);
    CHECK(Trit{+99}.value() == +1);
}

TEST_CASE("Trit negation") {
    CHECK((-T_POS) == T_NEG);
    CHECK((-T_NEG) == T_POS);
    CHECK((-T_ZERO) == T_ZERO);
}

TEST_CASE("trit_add covers all 9 cases") {
    struct Case { Trit a, b, sum, carry; };
    Case cases[] = {
        {T_NEG, T_NEG, T_POS, T_NEG},
        {T_NEG, T_ZERO, T_NEG, T_ZERO},
        {T_NEG, T_POS, T_ZERO, T_ZERO},
        {T_ZERO, T_NEG, T_NEG, T_ZERO},
        {T_ZERO, T_ZERO, T_ZERO, T_ZERO},
        {T_ZERO, T_POS, T_POS, T_ZERO},
        {T_POS, T_NEG, T_ZERO, T_ZERO},
        {T_POS, T_ZERO, T_POS, T_ZERO},
        {T_POS, T_POS, T_NEG, T_POS},
    };
    for (auto& c : cases) {
        auto r = trit_add(c.a, c.b);
        CHECK(r.sum == c.sum);
        CHECK(r.carry == c.carry);
    }
}

TEST_CASE("trit_full_add over Trit^3 reconstructs sum") {
    for (int a : {-1, 0, 1}) for (int b : {-1, 0, 1}) for (int c : {-1, 0, 1}) {
        auto r = trit_full_add(Trit{a}, Trit{b}, Trit{c});
        CHECK(r.sum.value() + 3 * r.carry.value() == a + b + c);
    }
}

TEST_CASE("trit_max and trit_min") {
    CHECK(trit_max(T_NEG, T_POS) == T_POS);
    CHECK(trit_min(T_NEG, T_POS) == T_NEG);
}
