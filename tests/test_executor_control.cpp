#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>

using namespace ter;

static void load(Sim& s, std::initializer_list<Instr> code) {
    size_t a = 0;
    for (const auto& i : code) s.mem().store_word(a++, encode(i));
}

TEST_CASE("TJUMP") {
    Sim s(64);
    load(s, {
        {Opcode::TLOADI, 1, 0, 0, 1},
        {Opcode::TJUMP,  0, 0, 0, 4},
        {Opcode::TLOADI, 1, 0, 0, 999},
        {Opcode::THALT,  0, 0, 0, 0},
        {Opcode::TLOADI, 2, 0, 0, 7},
        {Opcode::THALT,  0, 0, 0, 0},
    });
    s.run();
    CHECK(s.regs().read_scalar(1).to_int() == 1);
    CHECK(s.regs().read_scalar(2).to_int() == 7);
}

TEST_CASE("TBEQ taken") {
    Sim s(64);
    load(s, {
        {Opcode::TLOADI, 1, 0, 0, 5},
        {Opcode::TLOADI, 2, 0, 0, 5},
        {Opcode::TBEQ,   0, 1, 2, 5},
        {Opcode::TLOADI, 3, 0, 0, 999},
        {Opcode::THALT,  0, 0, 0, 0},
        {Opcode::TLOADI, 3, 0, 0, 1},
        {Opcode::THALT,  0, 0, 0, 0},
    });
    s.run();
    CHECK(s.regs().read_scalar(3).to_int() == 1);
}

TEST_CASE("TCALL/TRET round-trip") {
    Sim s(64);
    load(s, {
        {Opcode::TLOADI, 26, 0, 0, 50},
        {Opcode::TCALL,  0, 0, 0, 4},
        {Opcode::TLOADI, 1, 0, 0, 88},
        {Opcode::THALT,  0, 0, 0, 0},
        {Opcode::TLOADI, 2, 0, 0, 33},
        {Opcode::TRET,   0, 0, 0, 0},
    });
    s.run();
    CHECK(s.regs().read_scalar(1).to_int() == 88);
    CHECK(s.regs().read_scalar(2).to_int() == 33);
}
