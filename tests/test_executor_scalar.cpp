#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>

using namespace ter;

static Sim build_sim_with(std::initializer_list<Instr> code) {
    Sim s(64);
    size_t addr = 0;
    for (const auto& i : code) s.mem().store_word(addr++, encode(i));
    return s;
}

TEST_CASE("TLOADI") {
    Sim s = build_sim_with({
        {Opcode::TLOADI, 1, 0, 0, 42},
        {Opcode::THALT,  0, 0, 0, 0},
    });
    s.run();
    CHECK(s.regs().read_scalar(1).to_int() == 42);
}

TEST_CASE("TADD adds two registers") {
    Sim s = build_sim_with({
        {Opcode::TLOADI, 1, 0, 0, 10},
        {Opcode::TLOADI, 2, 0, 0, 20},
        {Opcode::TADD,   3, 1, 2, 0},
        {Opcode::THALT,  0, 0, 0, 0},
    });
    s.run();
    CHECK(s.regs().read_scalar(3).to_int() == 30);
}

TEST_CASE("TSUB and TNEG") {
    Sim s = build_sim_with({
        {Opcode::TLOADI, 1, 0, 0, 50},
        {Opcode::TLOADI, 2, 0, 0, 30},
        {Opcode::TSUB,   3, 1, 2, 0},
        {Opcode::TNEG,   4, 3, 0, 0},
        {Opcode::THALT,  0, 0, 0, 0},
    });
    s.run();
    CHECK(s.regs().read_scalar(3).to_int() == 20);
    CHECK(s.regs().read_scalar(4).to_int() == -20);
}

TEST_CASE("counters reflect ops") {
    Sim s = build_sim_with({
        {Opcode::TLOADI, 1, 0, 0, 7},
        {Opcode::TADD,   2, 1, 1, 0},
        {Opcode::TADD,   3, 2, 1, 0},
        {Opcode::THALT,  0, 0, 0, 0},
    });
    s.run();
    CHECK(s.counters().get(Opcode::TADD)   == 2);
    CHECK(s.counters().get(Opcode::TLOADI) == 1);
    CHECK(s.counters().get(Opcode::THALT)  == 1);
}
