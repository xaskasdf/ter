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

static float read_x_step() {
    std::ifstream meta("lut_data/sigmoid_lut.meta");
    REQUIRE(meta.is_open());
    std::string line;
    float x_step = 0.0f;
    while (std::getline(meta, line)) {
        if (line.rfind("x_step=", 0) == 0) x_step = std::stof(line.substr(7));
    }
    return x_step;
}

TEST_CASE("tk_silu matches numpy silu within bounded rel_err") {
    constexpr int N = 27;
    constexpr int N_ENTRIES = 256;
    constexpr int OUT_SCALE = 9841;

    std::vector<float> gate(N);
    std::mt19937 rng(0xDADA);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& v : gate) v = dist(rng);

    std::vector<float> y_ref(N);
    for (int i = 0; i < N; ++i) {
        double sig = 1.0 / (1.0 + std::exp(-double(gate[i])));
        y_ref[i] = static_cast<float>(double(gate[i]) * sig);
    }

    TritTensor gt = quantize(gate.data(), {N}, 9);

    auto sig_i32 = read_i32("lut_data/sigmoid_lut.bin");
    REQUIRE(sig_i32.size() == N_ENTRIES);
    std::vector<int> sigmoid_lut(sig_i32.begin(), sig_i32.end());

    float x_step = read_x_step();
    REQUIRE(x_step > 0.0f);

    Sim s(8192);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    KernelId id = kt.find("tk_silu");
    REQUIRE(id.valid);

    // Kernel code occupies addrs 0..~120; scratch hardcoded at 700-726.
    // Place data well clear: x@300, y@400, lut@1000.
    int x_addr = 300, y_addr = 400, lut_addr = 1000;
    for (int i = 0; i < N; ++i) s.mem().store_word(static_cast<size_t>(x_addr + i), gt.payload[i]);
    s.load_lut(lut_addr, sigmoid_lut);

    int64_t x_scale_div = std::max<int64_t>(1, static_cast<int64_t>(std::round(x_step / gt.scale)));

    std::vector<int64_t> args = {x_addr, y_addr, lut_addr, x_scale_div, 0, 0, 0};
    s.call_kernel(kt, id, args);

    std::vector<int> y_int(N);
    for (int i = 0; i < N; ++i) y_int[i] = static_cast<int>(s.mem().load_word(static_cast<size_t>(y_addr + i)).to_int());

    // Recovery: y_int = (gate / gt.scale) * (sigmoid * OUT_SCALE) → silu = y_int * gt.scale / OUT_SCALE
    float recovery = gt.scale / static_cast<float>(OUT_SCALE);
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
    CHECK(s.counters().get(Opcode::TLOAD) >= 54);
}

TEST_CASE("Full SwiGLU = silu(gate) * up via host composition") {
    constexpr int N = 27;
    constexpr int OUT_SCALE = 9841;

    std::vector<float> gate(N), up(N);
    std::mt19937 rng(0xFADE);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int i = 0; i < N; ++i) { gate[i] = dist(rng); up[i] = dist(rng); }

    std::vector<float> y_ref(N);
    for (int i = 0; i < N; ++i) {
        double sig = 1.0 / (1.0 + std::exp(-double(gate[i])));
        y_ref[i] = static_cast<float>(double(gate[i]) * sig * double(up[i]));
    }

    TritTensor gt = quantize(gate.data(), {N}, 9);

    auto sig_i32 = read_i32("lut_data/sigmoid_lut.bin");
    std::vector<int> sigmoid_lut(sig_i32.begin(), sig_i32.end());
    float x_step = read_x_step();

    Sim s(8192);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    KernelId id = kt.find("tk_silu");
    // Kernel code occupies addrs 0..~120; scratch hardcoded at 700-726.
    // Place data well clear: x@300, y@400, lut@1000.
    int x_addr = 300, y_addr = 400, lut_addr = 1000;
    for (int i = 0; i < N; ++i) s.mem().store_word(static_cast<size_t>(x_addr + i), gt.payload[i]);
    s.load_lut(lut_addr, sigmoid_lut);

    int64_t x_scale_div = std::max<int64_t>(1, static_cast<int64_t>(std::round(x_step / gt.scale)));
    std::vector<int64_t> args = {x_addr, y_addr, lut_addr, x_scale_div, 0, 0, 0};
    s.call_kernel(kt, id, args);

    float recovery = gt.scale / static_cast<float>(OUT_SCALE);
    std::vector<float> y(N);
    for (int i = 0; i < N; ++i) {
        float silu = static_cast<float>(s.mem().load_word(static_cast<size_t>(y_addr + i)).to_int()) * recovery;
        y[i] = silu * up[i];
    }

    double max_rel = 0.0;
    for (int i = 0; i < N; ++i) {
        double ref = y_ref[i];
        double got = y[i];
        double denom = std::max(1e-3, std::fabs(ref));
        double rel = std::fabs(got - ref) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    CHECK(max_rel < 1e-1);
}
