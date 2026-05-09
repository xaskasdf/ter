#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>

using namespace ter;

static void load(Sim& s, std::initializer_list<Instr> code) {
    size_t a = 0;
    for (const auto& i : code) s.mem().store_word(a++, encode(i));
}

TEST_CASE("TAND3 = per-trit min") {
    Sim s(32);
    load(s, {
        {Opcode::TLOADI, 1, 0, 0, 5},
        {Opcode::TLOADI, 2, 0, 0, -3},
        {Opcode::TAND3,  3, 1, 2, 0},
        {Opcode::THALT,  0, 0, 0, 0},
    });
    s.run();
    auto a = Word27::from_int(5);
    auto b = Word27::from_int(-3);
    auto r = s.regs().read_scalar(3);
    for (int i = 0; i < 27; ++i) {
        CHECK(r.trit(i) == trit_min(a.trit(i), b.trit(i)));
    }
}

TEST_CASE("TCMP returns sign of a-b") {
    Sim s(32);
    load(s, {
        {Opcode::TLOADI, 1, 0, 0, 10},
        {Opcode::TLOADI, 2, 0, 0, 20},
        {Opcode::TCMP,   3, 1, 2, 0},
        {Opcode::THALT,  0, 0, 0, 0},
    });
    s.run();
    CHECK(s.regs().read_scalar(3).trit(0) == T_NEG);
}
