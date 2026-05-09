#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>

using namespace ter;

TEST_CASE("TSTORE then TLOAD round-trip") {
    Sim s(64);
    Instr code[] = {
        {Opcode::TLOADI, 1, 0, 0, 999},
        {Opcode::TLOADI, 2, 0, 0, 50},
        {Opcode::TSTORE, 0, 1, 2, 0},
        {Opcode::TLOAD,  3, 2, 0, 0},
        {Opcode::THALT,  0, 0, 0, 0},
    };
    for (size_t i = 0; i < sizeof(code)/sizeof(code[0]); ++i)
        s.mem().store_word(i, encode(code[i]));
    s.run();
    CHECK(s.regs().read_scalar(3).to_int() == 999);
}
