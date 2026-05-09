#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/counters.hpp>

using namespace ter;

TEST_CASE("OpCounters bumps per opcode") {
    OpCounters c;
    CHECK(c.get(Opcode::TADD) == 0);
    c.bump(Opcode::TADD);
    c.bump(Opcode::TADD);
    c.bump(Opcode::TSUB);
    CHECK(c.get(Opcode::TADD) == 2);
    CHECK(c.get(Opcode::TSUB) == 1);
}

TEST_CASE("OpCounters total") {
    OpCounters c;
    c.bump(Opcode::TADD);
    c.bump(Opcode::THALT);
    CHECK(c.total() == 2);
}
