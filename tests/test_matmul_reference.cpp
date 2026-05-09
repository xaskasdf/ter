#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/vec.hpp>
#include <fstream>
#include <vector>
#include <algorithm>

using namespace ter;

static std::vector<int32_t> read_i32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.is_open());
    f.seekg(0, std::ios::end);
    size_t n = static_cast<size_t>(f.tellg()) / 4;
    f.seekg(0);
    std::vector<int32_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4));
    return v;
}

static std::vector<int64_t> read_i64(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.is_open());
    f.seekg(0, std::ios::end);
    size_t n = static_cast<size_t>(f.tellg()) / 8;
    f.seekg(0);
    std::vector<int64_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 8));
    return v;
}

TEST_CASE("64x64 matmul via host-driven TVMAC matches numpy") {
    constexpr int N = 64;
    auto A = read_i32("matmul_data/A.bin");
    auto B = read_i32("matmul_data/B.bin");
    auto C_ref = read_i64("matmul_data/C.bin");
    REQUIRE(static_cast<int>(A.size()) == N * N);

    std::vector<int64_t> C(N * N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int64_t sum = 0;
            for (int k0 = 0; k0 < N; k0 += Vec::kLanes) {
                Vec va, vb;
                int chunk = std::min<int>(Vec::kLanes, N - k0);
                for (int t = 0; t < Vec::kLanes; ++t) {
                    va.set_lane(t, t < chunk ? A[i * N + (k0 + t)] : 0);
                    vb.set_lane(t, t < chunk ? B[(k0 + t) * N + j] : 0);
                }
                VAccum acc;
                vec_mac(acc, va, vb);
                sum += vec_sum(acc);
            }
            C[i * N + j] = sum;
        }
    }

    for (int idx = 0; idx < N * N; ++idx) {
        CHECK(C[idx] == C_ref[idx]);
    }
}
