#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/assembler.hpp>

using namespace ter;

TEST_CASE("assemble SIMD instructions") {
    auto blob = assemble(R"(
        tvbroadcast v0, 3
        tvbroadcast v1, 4
        tvmac       a0, v0, v1
        tvsum       r1, a0
        thalt
    )");
    REQUIRE(blob.size() == 5);
    CHECK(decode(blob[0]).op == Opcode::TVBROADCAST);
    CHECK(decode(blob[2]).op == Opcode::TVMAC);
    CHECK(decode(blob[3]).op == Opcode::TVSUM);
}
