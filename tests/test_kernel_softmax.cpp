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

TEST_CASE("tk_softmax matches numpy reference within bounded rel_err") {
    constexpr int N = 27;
    constexpr int N_ENTRIES = 256;
    constexpr int OUT_SCALE = 9841;

    std::vector<float> x(N);
    std::mt19937 rng(0xCAFE);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& v : x) v = dist(rng);

    std::vector<double> ex(N);
    double sum_e = 0.0;
    for (int i = 0; i < N; ++i) { ex[i] = std::exp(double(x[i])); sum_e += ex[i]; }
    std::vector<float> y_ref(N);
    for (int i = 0; i < N; ++i) y_ref[i] = static_cast<float>(ex[i] / sum_e);

    TritTensor xt = quantize(x.data(), {N}, 9);

    auto exp_i32 = read_i32("lut_data/exp_lut.bin");
    auto rcp_i32 = read_i32("lut_data/rcp_lut.bin");
    REQUIRE(exp_i32.size() == N_ENTRIES);
    REQUIRE(rcp_i32.size() == N_ENTRIES);

    std::vector<int> exp_lut(exp_i32.begin(), exp_i32.end());
    std::vector<int> rcp_lut(rcp_i32.begin(), rcp_i32.end());

    float exp_max = 0.0f, rcp_max = 0.0f, x_step = 0.0f;
    {
        std::ifstream meta("lut_data/softmax_lut.meta");
        REQUIRE(meta.is_open());
        std::string line;
        while (std::getline(meta, line)) {
            if (line.rfind("exp_max=", 0) == 0) exp_max = std::stof(line.substr(8));
            else if (line.rfind("rcp_max=", 0) == 0) rcp_max = std::stof(line.substr(8));
            else if (line.rfind("x_step=", 0) == 0) x_step = std::stof(line.substr(7));
        }
    }
    REQUIRE(exp_max > 0.0f);
    REQUIRE(rcp_max > 0.0f);
    REQUIRE(x_step > 0.0f);

    Sim s(8192);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    KernelId id = kt.find("tk_softmax");
    REQUIRE(id.valid);

    int x_addr = 100, y_addr = 200, exp_lut_addr = 1000, rcp_lut_addr = 2000;
    for (int i = 0; i < N; ++i) s.mem().store_word(static_cast<size_t>(x_addr + i), ter::Word27::from_int(xt.payload[i]));
    s.load_lut(exp_lut_addr, exp_lut);
    s.load_lut(rcp_lut_addr, rcp_lut);

    int64_t x_scale_div = static_cast<int64_t>(std::round(x_step / xt.scale));
    if (x_scale_div < 1) x_scale_div = 1;

    int64_t sum_div = (N * static_cast<int64_t>(OUT_SCALE)) / 255;

    std::vector<int64_t> args = {x_addr, y_addr, exp_lut_addr, rcp_lut_addr,
                                 x_scale_div, sum_div, 0};
    s.call_kernel(kt, id, args);

    std::vector<int> y_int(N);
    for (int i = 0; i < N; ++i) y_int[i] = static_cast<int>(s.mem().load_word(static_cast<size_t>(y_addr + i)).to_int());

    // Recovery derivation:
    //   exp_lut[ei] ≈ exp(x[i]) * OUT_SCALE / exp_max
    //   sum_int      ≈ sum_exp   * OUT_SCALE / exp_max
    //   rcp_idx      ≈ sum_int / sum_div  ≈ sum_exp * 255 / (exp_max * N)
    //   rcp_lut[ri]  ≈ OUT_SCALE / (rcp_idx+1) ≈ OUT_SCALE * exp_max * N / (sum_exp * 255)
    //   y_int[i]     ≈ exp(x[i])/sum_exp * OUT_SCALE^2 * N / 255
    //   → recovery   = 255 / (OUT_SCALE^2 * N)
    float recovery = 255.0f / (static_cast<float>(OUT_SCALE) * static_cast<float>(OUT_SCALE) * static_cast<float>(N));
    std::vector<float> y(N);
    for (int i = 0; i < N; ++i) y[i] = static_cast<float>(y_int[i]) * recovery;

    double max_rel = 0.0;
    for (int i = 0; i < N; ++i) {
        double ref = y_ref[i];
        double got = y[i];
        double denom = std::max(1e-3, std::fabs(ref));
        double rel = std::fabs(got - ref) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    CHECK(max_rel < 1e-1);

    CHECK(s.counters().get(Opcode::TVMUL) == 1);
    CHECK(s.counters().get(Opcode::TVLOAD) == 2);
    CHECK(s.counters().get(Opcode::TVSTORE) == 1);
    CHECK(s.counters().get(Opcode::TLOAD) >= 55);
}
