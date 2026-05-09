#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/numfmt.hpp>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <string>

#ifndef TER_KERNELS_DIR
#error "TER_KERNELS_DIR must be defined"
#endif

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

TEST_CASE("tk_rmsnorm matches numpy reference within bounded rel_err") {
    constexpr int N = 27;
    constexpr int N_ENTRIES = 256;
    constexpr int OUT_SCALE = 9841;

    std::vector<float> x(N);
    std::mt19937 rng(0xBEEF);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : x) v = dist(rng);

    constexpr float eps = 1e-6f;
    double sum_sq = 0.0;
    for (int i = 0; i < N; ++i) sum_sq += double(x[i]) * double(x[i]);
    double mean_sq = sum_sq / N;
    double rsqrt_ref = 1.0 / std::sqrt(mean_sq + eps);
    std::vector<float> y_ref(N);
    for (int i = 0; i < N; ++i) y_ref[i] = static_cast<float>(double(x[i]) * rsqrt_ref);

    TritTensor xt = quantize(x.data(), {N}, 9);

    auto lut_i32 = read_i32("lut_data/rsqrt_lut.bin");
    REQUIRE(lut_i32.size() == N_ENTRIES);
    std::vector<int> lut(lut_i32.begin(), lut_i32.end());

    Sim s(4096);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    KernelId id = kt.find("tk_rmsnorm");
    REQUIRE(id.valid);

    int x_addr = 100, y_addr = 200, lut_addr = 1000;
    for (int i = 0; i < N; ++i) s.mem().store_word(static_cast<size_t>(x_addr + i), ter::Word27::from_int(xt.payload[i]));
    s.load_lut(lut_addr, lut);

    int64_t mti = 9841;
    int64_t sum_div = (N * mti * mti) / 255;

    std::vector<int64_t> args = {x_addr, y_addr, lut_addr, sum_div, 255, 0, 0};
    s.call_kernel(kt, id, args);

    std::vector<int> y_int(N);
    for (int i = 0; i < N; ++i) {
        y_int[i] = static_cast<int>(s.mem().load_word(static_cast<size_t>(y_addr + i)).to_int());
    }

    // Read rsq_max from LUT meta side-channel for the recovery scale.
    float rsq_max = 0.0f;
    {
        std::ifstream meta("lut_data/rsqrt_lut.meta");
        REQUIRE(meta.is_open());
        std::string line;
        while (std::getline(meta, line)) {
            auto pos = line.find("rsq_max=");
            if (pos != std::string::npos) {
                rsq_max = std::stof(line.substr(pos + 8));
            }
        }
    }
    REQUIRE(rsq_max > 0.0f);

    // Recovery formula derivation:
    // TVMUL produces y_int[i] = v0[i] * lut[idx] (non-saturating wide multiply).
    // lut[idx] ≈ OUT_SCALE * rsqrt(vals[idx]) / rsq_max
    //          ≈ OUT_SCALE * mti * sqrt(N / sum_sq_int) / rsq_max
    // We want y[i] = v0[i] * sqrt(N / sum_sq_int), so the recovery factor that
    // cancels (OUT_SCALE * mti / rsq_max) is:
    //   recovery = rsq_max / (OUT_SCALE * mti)
    std::vector<float> y(N);
    float recovery = rsq_max / (static_cast<float>(OUT_SCALE) * static_cast<float>(mti));
    for (int i = 0; i < N; ++i) y[i] = static_cast<float>(y_int[i]) * recovery;

    double max_rel = 0.0;
    for (int i = 0; i < N; ++i) {
        double ref = y_ref[i];
        double got = y[i];
        double denom = std::max(1.0, std::fabs(ref));
        double rel = std::fabs(got - ref) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    CHECK(max_rel < 5e-2);

    CHECK(s.counters().get(Opcode::TVMAC) == 1);
    CHECK(s.counters().get(Opcode::TVSUM) == 1);
    CHECK(s.counters().get(Opcode::TVMUL) == 1);
}
