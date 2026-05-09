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
