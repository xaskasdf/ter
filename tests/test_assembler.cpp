#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/assembler.hpp>

using namespace ter;

TEST_CASE("assemble single instruction") {
    auto blob = assemble("tloadi r1, 42\n");
    REQUIRE(blob.size() == 1);
    auto i = decode(blob[0]);
    CHECK(i.op == Opcode::TLOADI);
    CHECK(i.dst == 1);
    CHECK(i.imm == 42);
}

TEST_CASE("assemble multi-line with comments") {
    auto blob = assemble(R"(
        ; load 5 into r1
        tloadi r1, 5

        tloadi r2, 7
        tadd   r3, r1, r2
        thalt
    )");
    REQUIRE(blob.size() == 4);
    CHECK(decode(blob[0]).op == Opcode::TLOADI);
    CHECK(decode(blob[2]).op == Opcode::TADD);
    CHECK(decode(blob[3]).op == Opcode::THALT);
}

TEST_CASE("labels resolve to instruction addresses") {
    auto blob = assemble(R"(
        tloadi r1, 0
    loop:
        tadd   r1, r1, r1
        tjump  loop
    )");
    REQUIRE(blob.size() == 3);
    auto jmp = decode(blob[2]);
    CHECK(jmp.op == Opcode::TJUMP);
    CHECK(jmp.imm == 1);
}

TEST_CASE("error on unknown mnemonic") {
    CHECK_THROWS_AS(assemble("tnope r1, r2\n"), AssemblerError);
}
