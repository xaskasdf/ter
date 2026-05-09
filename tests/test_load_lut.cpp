#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/assembler.hpp>
#include <vector>

using namespace ter;

TEST_CASE("load_lut writes int values as Word27s and a kernel reads them back") {
    auto blob = assemble(R"(
        tload r1, r1
        thalt
    )");
    Sim s(2048);
    KernelTable kt;
    KernelId id = kt.install(s, "lut_read", blob);

    std::vector<int> lut = {10, 20, -5, 99, 0, 1234};
    int lut_addr = 100;
    s.load_lut(lut_addr, lut);

    for (int i = 0; i < static_cast<int>(lut.size()); ++i) {
        std::vector<int64_t> args = {lut_addr + i, 0, 0, 0, 0, 0, 0};
        int64_t r = s.call_kernel(kt, id, args);
        CHECK(r == lut[i]);
    }
}
