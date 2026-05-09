#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/numfmt.hpp>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

#ifndef TER_KERNELS_DIR
#error "TER_KERNELS_DIR must be defined by the build system"
#endif

using namespace ter;

static std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.is_open());
    f.seekg(0, std::ios::end);
    size_t n = static_cast<size_t>(f.tellg()) / 4;
    f.seekg(0);
    std::vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4));
    return v;
}

TEST_CASE("format-B matmul via kernel matches numpy within bounded rel_err") {
    int M, N, K;
    {
        std::ifstream f("matmul_b_data/shape.txt"); REQUIRE(f.is_open());
        f >> M >> N >> K;
    }
    auto A = read_f32("matmul_b_data/A.bin");
    auto B = read_f32("matmul_b_data/B.bin");
    auto C_ref = read_f32("matmul_b_data/C.bin");

    TritTensor At = quantize(A.data(), {M, K}, 9);
    TritTensor Bt = quantize(B.data(), {K, N}, 9);

    Sim s(16 * 1024);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    KernelId id = kt.find("tk_matmul_b_9t");
    REQUIRE(id.valid);

    int x_base = 1000, w_base = 4000, y_addr = 7000;
    int scratch_x = 9000, scratch_w = 9100;

    for (int idx = 0; idx < M * K; ++idx) s.mem().store_word(static_cast<size_t>(x_base + idx), ter::Word27::from_int(At.payload[idx]));
    for (int idx = 0; idx < K * N; ++idx) s.mem().store_word(static_cast<size_t>(w_base + idx), ter::Word27::from_int(Bt.payload[idx]));

    std::vector<float> C(M * N);
    int tiles = (K + 26) / 27;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int64_t int_acc = 0;
            for (int k0 = 0; k0 < K; k0 += 27) {
                int chunk = std::min(27, K - k0);

                // Always copy through scratch buffers: ensures the W column
                // is contiguous and X chunk is contiguous (for the next tile).
                for (int t = 0; t < 27; ++t) {
                    int xv = (t < chunk) ? At.payload[i * K + k0 + t] : 0;
                    int wv = (t < chunk) ? Bt.payload[(k0 + t) * N + j] : 0;
                    s.mem().store_word(static_cast<size_t>(scratch_x + t), Word27::from_int(xv));
                    s.mem().store_word(static_cast<size_t>(scratch_w + t), Word27::from_int(wv));
                }

                std::vector<int64_t> args = {scratch_x, scratch_w, y_addr, 0, 0, 0, 0};
                s.call_kernel(kt, id, args);
                int_acc += s.mem().load_word(static_cast<size_t>(y_addr)).to_int();
            }
            C[i * N + j] = static_cast<float>(int_acc) * At.scale * Bt.scale;
        }
    }

    double max_rel = 0.0;
    for (int idx = 0; idx < M * N; ++idx) {
        double ref = static_cast<double>(C_ref[idx]);
        double got = static_cast<double>(C[idx]);
        double denom = std::max(1.0, std::fabs(ref));
        double rel = std::fabs(got - ref) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    CHECK(max_rel < 1e-2);

    CHECK(s.counters().get(Opcode::TVMAC) == static_cast<uint64_t>(M * N * tiles));
}
