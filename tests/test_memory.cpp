#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/memory.hpp>

using namespace ter;

TEST_CASE("Memory store/load round-trip") {
    Memory mem(1024);
    auto w = Word27::from_int(424242);
    mem.store_word(100, w);
    CHECK(mem.load_word(100) == w);
}

TEST_CASE("Memory zero-initialised") {
    Memory mem(16);
    CHECK(mem.load_word(0).to_int() == 0);
    CHECK(mem.load_word(15).to_int() == 0);
}

TEST_CASE("Memory bounds violation throws") {
    Memory mem(8);
    CHECK_THROWS_AS(mem.load_word(8), std::out_of_range);
    CHECK_THROWS_AS(mem.store_word(99, Word27{}), std::out_of_range);
}
