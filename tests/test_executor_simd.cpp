#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>

using namespace ter;

static void load(Sim& s, std::initializer_list<Instr> code) {
    size_t a = 0;
    for (const auto& i : code) s.mem().store_word(a++, encode(i));
}

TEST_CASE("TVBROADCAST + TVADD lane-wise") {
    Sim s(64);
    load(s, {
        {Opcode::TVBROADCAST, 0, 0, 0, 5},
        {Opcode::TVBROADCAST, 1, 0, 0, 7},
        {Opcode::TVADD,       2, 0, 1, 0},
        {Opcode::THALT,       0, 0, 0, 0},
    });
    s.run();
    auto v2 = s.regs().read_vec(2);
    for (int i = 0; i < Vec::kLanes; ++i) CHECK(v2.lane(i) == 12);
}

TEST_CASE("TVMAC accumulates and TVSUM reduces") {
    Sim s(64);
    load(s, {
        {Opcode::TVBROADCAST, 0, 0, 0, 3},
        {Opcode::TVBROADCAST, 1, 0, 0, 4},
        {Opcode::TVMAC,       0, 0, 1, 0},
        {Opcode::TVMAC,       0, 0, 1, 0},
        {Opcode::TVSUM,       1, 0, 0, 0},
        {Opcode::THALT,       0, 0, 0, 0},
    });
    s.run();
    CHECK(s.regs().read_scalar(1).to_int() == 27 * 24);
    CHECK(s.counters().get(Opcode::TVMAC) == 2);
}
