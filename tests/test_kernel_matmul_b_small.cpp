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

TEST_CASE("multi-tile matmul: 1x54 @ 54x1 via two tiles") {
    Sim s(2048);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    KernelId id = kt.find("tk_matmul_b_9t");

    constexpr int K = 54;
    std::vector<int> X(K), W(K);
    for (int i = 0; i < K; ++i) { X[i] = (i * 7) % 19 - 9; W[i] = (i * 5) % 13 - 6; }

    int xa = 100, wa = 1000, ya = 2000;
    for (int i = 0; i < K; ++i) {
        s.mem().store_word(static_cast<size_t>(xa + i), Word27::from_int(X[i]));
        s.mem().store_word(static_cast<size_t>(wa + i), Word27::from_int(W[i]));
    }

    int64_t expected = 0;
    for (int i = 0; i < K; ++i) expected += int64_t{X[i]} * int64_t{W[i]};

    // a0 is a persistent accumulator: each call adds its tile's dot product into a0.
    // After both tiles, tvsum writes the full sum to ya — no host-side addition needed.
    for (int t = 0; t < 2; ++t) {
        int tile_x = xa + t * 27;
        int tile_w = wa + t * 27;
        std::vector<int64_t> args = {tile_x, tile_w, ya, 0, 0, 0, 0};
        s.call_kernel(kt, id, args);
    }
    int64_t got = s.mem().load_word(static_cast<size_t>(ya)).to_int();

    CHECK(got == expected);
    CHECK(s.counters().get(Opcode::TVMAC) == 2);
}
