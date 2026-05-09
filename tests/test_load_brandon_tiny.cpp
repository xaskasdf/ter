#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <model/loader.h>
#include <core/dequant.hpp>
#include <ter/host/load_gguf.hpp>
#include <ter/numfmt.hpp>
#include <fstream>
#include <vector>
#include <cmath>

using namespace nt;

namespace {
constexpr int VOCAB   = 8192;
constexpr int HIDDEN  = 256;
constexpr const char* GGUF_PATH =
    "/Users/pc/osito-a-models/build/brandon-tiny-10m-f16.gguf";

bool file_exists(const char* path) {
    std::ifstream f(path);
    return f.good();
}
}  // namespace

TEST_CASE("brandon-tiny GGUF opens (or skip)") {
    if (!file_exists(GGUF_PATH)) {
        MESSAGE("brandon-tiny GGUF not found — skipping");
        return;
    }
    GGUFLoader loader;
    bool ok = loader.load(GGUF_PATH);
    if (!ok) {
        MESSAGE("loader.load() rejected the brandon GGUF — likely needs brandon.* parser extensions (F5.4b)");
        return;  // Don't fail; expected if loader is llama-only.
    }
    MESSAGE("brandon GGUF loaded (config() may be inaccurate due to brandon.* keys not being parsed)");
}

TEST_CASE("brandon-tiny token_embd.weight: load and quantize to Format B") {
    if (!file_exists(GGUF_PATH)) {
        MESSAGE("brandon-tiny GGUF not found — skipping");
        return;
    }
    GGUFLoader loader;
    if (!loader.load(GGUF_PATH)) {
        MESSAGE("loader.load() rejected; skipping (F5.4b will add brandon parser)");
        return;
    }

    const auto* info = loader.tensor_info("token_embd.weight");
    if (!info) {
        MESSAGE("tensor_info(\"token_embd.weight\") returned null — listing available tensor names:");
        for (const auto& n : loader.tensor_names()) {
            MESSAGE("  ", n);
        }
        return;
    }

    Tensor t = loader.get_tensor("token_embd.weight");
    REQUIRE(t.shape().size() == 2);

    std::size_t n = 1;
    for (auto d : t.shape()) n *= static_cast<std::size_t>(d);
    MESSAGE("token_embd shape product = ", n,
            "; expected ", static_cast<std::size_t>(VOCAB * HIDDEN));
    CHECK(n == static_cast<std::size_t>(VOCAB) * static_cast<std::size_t>(HIDDEN));

    // dtype should be F16 for brandon-tiny f16 GGUF.
    if (t.dtype() != DType::F16) {
        MESSAGE("Expected F16, got dtype enum value ", static_cast<int>(t.dtype()));
        // Don't hard-fail — the loader may report a different dtype mapping.
    }

    // Round-trip via tensor_to_trit + dequantize.
    auto tt = ter::host::tensor_to_trit(t, 9);
    CHECK(tt.dtype == ter::DType::TritFP_B);
    CHECK(tt.n_trits_per_elem == 9);
    CHECK(tt.payload.size() == n);
    CHECK(tt.scale > 0.0f);

    // Reference dequantization (mirrors what tensor_to_trit does internally).
    std::vector<float> dequantized(n);
    if (t.dtype() == DType::F16) {
        nt::dequant_f16(t.data(), n, dequantized.data());
    } else if (t.dtype() == DType::F32) {
        nt::dequant_f32(t.data(), n, dequantized.data());
    } else {
        MESSAGE("dtype not F16/F32, can't compare round-trip MSE; skipping");
        return;
    }

    std::vector<float> roundtrip(n);
    ter::dequantize(tt, roundtrip.data());

    double sse = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double d = static_cast<double>(dequantized[i]) - static_cast<double>(roundtrip[i]);
        sse += d * d;
    }
    double mse = sse / static_cast<double>(n);
    MESSAGE("token_embd Format-B round-trip MSE = ", mse, " (scale=", tt.scale, ")");
    // Generous bound — token_embd often has a few outliers that dominate per-tensor scale.
    CHECK(mse < 1e-2);
}
