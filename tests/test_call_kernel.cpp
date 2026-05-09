#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/assembler.hpp>

using namespace ter;

TEST_CASE("call_kernel runs assembled program with args in R1..R7") {
    auto blob = assemble(R"(
        tadd r1, r1, r2
        thalt
    )");
    Sim s(256);
    KernelTable kt;
    KernelId id = kt.install(s, "test_add", blob);

    std::vector<int64_t> args = {10, 32, 0, 0, 0, 0, 0};
    int64_t r1 = s.call_kernel(kt, id, args);
    CHECK(r1 == 42);
}

TEST_CASE("call_kernel preserves OpCounters across invocations") {
    auto blob = assemble(R"(
        tadd r1, r1, r2
        thalt
    )");
    Sim s(256);
    KernelTable kt;
    KernelId id = kt.install(s, "addk", blob);

    std::vector<int64_t> a1 = {1, 1, 0, 0, 0, 0, 0};
    s.call_kernel(kt, id, a1);
    s.call_kernel(kt, id, a1);

    CHECK(s.counters().get(Opcode::TADD)  == 2);
    CHECK(s.counters().get(Opcode::THALT) == 2);
}

TEST_CASE("call_kernel resets caller-saved registers between invocations") {
    auto blob = assemble(R"(
        tadd r1, r1, r8
        thalt
    )");
    Sim s(256);
    KernelTable kt;
    KernelId id = kt.install(s, "addr8", blob);

    // First call: r1=10, r8 should be 0 (fresh) → result 10.
    std::vector<int64_t> a1 = {10, 0, 0, 0, 0, 0, 0};
    int64_t r = s.call_kernel(kt, id, a1);
    CHECK(r == 10);

    // Set R8 manually between calls (simulating leftover state).
    s.regs().write_scalar(8, Word27::from_int(99));

    // Second call: r1=10 again, r8 should be reset to 0 (not 99) → result 10.
    int64_t r2 = s.call_kernel(kt, id, a1);
    CHECK(r2 == 10);
}

TEST_CASE("call_kernel resets accumulator A0 between invocations") {
    auto blob = assemble(R"(
        tvbroadcast v0, 1
        tvmac       a0, v0, v0
        tvsum       r1, a0
        thalt
    )");
    Sim s(256);
    KernelTable kt;
    KernelId id = kt.install(s, "macsum", blob);

    std::vector<int64_t> args = {0, 0, 0, 0, 0, 0, 0};
    int64_t r = s.call_kernel(kt, id, args);
    CHECK(r == 27);  // 27 lanes of 1*1

    int64_t r2 = s.call_kernel(kt, id, args);
    CHECK(r2 == 27);  // A0 reset between calls; not 54
}
