#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/numfmt.hpp>
#include <vector>

#ifndef TER_KERNELS_DIR
#error "TER_KERNELS_DIR must be defined by the build system"
#endif

using namespace ter;

TEST_CASE("tk_matmul_b_9t computes a single 27-length dot product") {
    Sim s(1024);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    KernelId id = kt.find("tk_matmul_b_9t");
    REQUIRE(id.valid);

    constexpr int K = 27;
    std::vector<int> X(K), W(K);
    for (int i = 0; i < K; ++i) { X[i] = i - 13; W[i] = (i % 5) - 2; }
    for (int i = 0; i < K; ++i) {
        s.mem().store_word(200 + i, Word27::from_int(X[i]));
        s.mem().store_word(300 + i, Word27::from_int(W[i]));
    }
    int Y_addr = 500;

    int64_t expected = 0;
    for (int i = 0; i < K; ++i) expected += int64_t{X[i]} * int64_t{W[i]};

    std::vector<int64_t> args = {200, 300, Y_addr, 0, 0, 0, 0};
    s.call_kernel(kt, id, args);

    int64_t got = s.mem().load_word(static_cast<size_t>(Y_addr)).to_int();
    CHECK(got == expected);

    CHECK(s.counters().get(Opcode::TVMAC) == 1);
    CHECK(s.counters().get(Opcode::TVSUM) == 1);
}
