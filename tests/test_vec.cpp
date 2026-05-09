#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/vec.hpp>

using namespace ter;

TEST_CASE("Vec lane round-trip") {
    Vec v;
    for (int i = 0; i < Vec::kLanes; ++i) v.set_lane(i, i - 13);
    for (int i = 0; i < Vec::kLanes; ++i) CHECK(v.lane(i) == i - 13);
}

TEST_CASE("Vec lane add") {
    Vec a, b;
    for (int i = 0; i < Vec::kLanes; ++i) {
        a.set_lane(i, 100 + i);
        b.set_lane(i, -50 + i);
    }
    Vec r = vec_add(a, b);
    for (int i = 0; i < Vec::kLanes; ++i) CHECK(r.lane(i) == 50 + 2 * i);
}

TEST_CASE("Vec broadcast") {
    Vec v = vec_broadcast(7);
    for (int i = 0; i < Vec::kLanes; ++i) CHECK(v.lane(i) == 7);
}

TEST_CASE("vec_mac and vec_sum") {
    Vec a = vec_broadcast(3), b = vec_broadcast(4);
    VAccum acc;
    vec_mac(acc, a, b);
    vec_mac(acc, a, b);
    CHECK(vec_sum(acc) == 27 * 24);
}
