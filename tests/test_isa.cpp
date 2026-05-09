#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/isa.hpp>

using namespace ter;

TEST_CASE("encode/decode round-trip TADD") {
    Instr i; i.op = Opcode::TADD; i.dst=1; i.src1=2; i.src2=3; i.imm=0;
    Word27 w = encode(i);
    Instr d = decode(w);
    CHECK(d.op == i.op);
    CHECK(d.dst == 1);
    CHECK(d.src1 == 2);
    CHECK(d.src2 == 3);
    CHECK(d.imm == 0);
}

TEST_CASE("encode/decode TLOADI with negative imm") {
    Instr i; i.op = Opcode::TLOADI; i.dst=5; i.src1=0; i.src2=0; i.imm=-12345;
    Word27 w = encode(i);
    Instr d = decode(w);
    CHECK(d.op == Opcode::TLOADI);
    CHECK(d.dst == 5);
    CHECK(d.imm == -12345);
}

TEST_CASE("decode rejects unmapped opcode value") {
    Word27 w;
    for (int t = 21; t < 27; ++t) w.set_trit(t, T_POS);
    CHECK_THROWS_AS(decode(w), IllegalOpcode);
}
