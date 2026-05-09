#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/assembler.hpp>
#include <vector>

using namespace ter;

TEST_CASE("jump immediates relocate when kernel is installed at non-zero address") {
    // A kernel with both forward and backward branches that fails silently
    // (infinite loop or wrong result) if jump targets aren't shifted by entry_addr.
    //
    // Program: count r1 down to zero, return how many iterations (in r2).
    auto blob = assemble(R"(
        tloadi r2, 0
        tloadi r3, 1
loop:
        tbeq   r1, r0, done
        tsub   r1, r1, r3
        tadd   r2, r2, r3
        tjump  loop
done:
        tadd   r1, r2, r0
        thalt
    )");

    Sim s(2048);
    KernelTable kt;

    // First, install a "padding" kernel to push the next install address > 0.
    auto pad = assemble("thalt\n");
    KernelId pad_id = kt.install(s, "pad1", pad);
    REQUIRE(pad_id.valid);
    REQUIRE(pad_id.entry_addr == 0);

    // Now install the real kernel at the next free address.
    KernelId id = kt.install(s, "countdown", blob);
    REQUIRE(id.valid);
    REQUIRE(id.entry_addr > 0);   // critical: must be relocated

    std::vector<int64_t> args = {7, 0, 0, 0, 0, 0, 0};
    int64_t r = s.call_kernel(kt, id, args);
    CHECK(r == 7);   // 7 iterations from 7 down to 0

    // And once more with a different start to confirm reuse:
    std::vector<int64_t> args2 = {3, 0, 0, 0, 0, 0, 0};
    int64_t r2 = s.call_kernel(kt, id, args2);
    CHECK(r2 == 3);
}

TEST_CASE("kernel installed at non-zero address with TCALL inside relocates correctly") {
    // call_inside: pushes ret via tcall, returns immediately.
    auto blob = assemble(R"(
        tloadi r26, 100
        tcall  sub
        tadd   r1, r1, r2
        thalt
sub:
        tloadi r2, 99
        tret
    )");

    Sim s(2048);
    KernelTable kt;
    auto pad = assemble("thalt\n");
    kt.install(s, "pad1", pad);
    kt.install(s, "pad2", pad);   // Push entry_addr further from zero.
    KernelId id = kt.install(s, "calltest", blob);
    REQUIRE(id.entry_addr >= 2);

    std::vector<int64_t> args = {5, 0, 0, 0, 0, 0, 0};
    int64_t r = s.call_kernel(kt, id, args);
    CHECK(r == 104);  // r1=5, sub sets r2=99, then r1 = r1 + r2 = 104
}
