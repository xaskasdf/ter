// F12.x quality: real-weight ternary quantization quality on Llama 3.2 1B.
// Loads the Q8_0 GGUF, runs the SAME forward twice:
//   (1) baseline: Format B 9-trit (high precision, our reference)
//   (2) bitnet:   weights re-quantized to {-1, 0, +1} via absmean (BitNet b1.58)
// Reports argmax preservation, top-K overlap, and logit-space RMSE.
//
// Hypothesis: post-training BitNet quantization on a Llama-trained model
// likely degrades quality significantly (the model wasn't trained that way),
// but quantifies HOW MUCH. Useful baseline for the paper's H3 discussion.
//
// Runtime: 2 forwards x ~25s on Mac AVX2 = ~1 minute total.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/tx/transformer.hpp>
#include <ter/tx/lut_setup.hpp>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <model/loader.h>
#include <inference/sampler.h>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <vector>

#ifndef TER_KERNELS_DIR
#error "TER_KERNELS_DIR must be defined"
#endif

namespace {
constexpr const char* GGUF_PATH =
    "/Users/pc/osito-a-models/downloads/llama-3.2-1b-instruct/llama-3.2-1b-instruct-q8_0.gguf";

bool file_exists(const char* path) {
    std::ifstream f(path);
    return f.good();
}

struct Run {
    int argmax_id = -1;
    float argmax_logit = 0.0f;
    std::vector<int> top5;
    std::vector<float> logits;
};

Run run_one(bool bitnet) {
    using namespace ter;
    using namespace ter::tx;

    nt::GGUFLoader loader;
    REQUIRE(loader.load(GGUF_PATH));
    BrandonTransformer tx = load_llama_transformer(
        loader, /*max_seq=*/8, /*n_trits=*/9,
        /*format_a=*/false, /*format_a_mant_trits=*/9,
        /*bitnet_roundtrip=*/bitnet);

    Sim s(64 * 1024);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    LutAddrs luts = load_default_luts(s, "lut_data");

    Run r;
    int bos = loader.config().bos_token_id > 0 ? loader.config().bos_token_id : 128000;
    if (bos >= tx.vocab_size) bos = 128000;
    r.logits = forward_token(s, kt, tx, bos, /*pos=*/0, luts, /*state=*/nullptr);
    r.argmax_id = nt::Sampler::argmax(r.logits.data(), tx.vocab_size);
    r.argmax_logit = r.logits[static_cast<size_t>(r.argmax_id)];

    // Compute top-5 indices
    std::vector<std::pair<float, int>> idx(r.logits.size());
    for (size_t i = 0; i < r.logits.size(); ++i) idx[i] = {r.logits[i], static_cast<int>(i)};
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
        [](auto& a, auto& b) { return a.first > b.first; });
    r.top5.reserve(5);
    for (int i = 0; i < 5; ++i) r.top5.push_back(idx[i].second);
    return r;
}
}  // namespace

TEST_CASE("Llama 3.2 1B real-weight ternary quantization quality") {
    if (!file_exists(GGUF_PATH)) {
        MESSAGE("Llama 3.2 1B GGUF not found -- skipping");
        return;
    }

    Run baseline = run_one(/*bitnet=*/false);
    Run bitnet   = run_one(/*bitnet=*/true);

    REQUIRE(baseline.logits.size() == bitnet.logits.size());

    double sum_sq = 0.0, sum_abs = 0.0;
    for (size_t i = 0; i < baseline.logits.size(); ++i) {
        double d = baseline.logits[i] - bitnet.logits[i];
        sum_sq  += d * d;
        sum_abs += std::fabs(d);
    }
    double rmse = std::sqrt(sum_sq / static_cast<double>(baseline.logits.size()));
    double mae  = sum_abs / static_cast<double>(baseline.logits.size());

    int top5_overlap = 0;
    for (int b : baseline.top5)
        for (int q : bitnet.top5)
            if (b == q) ++top5_overlap;

    std::fprintf(stderr,
        "\n=== Llama 3.2 1B real-weight quantization quality (BOS forward) ===\n"
        "  baseline (B 9-trit) : argmax=%d  logit=%.4f  top5=[%d %d %d %d %d]\n"
        "  bitnet  ({-1,0,+1}) : argmax=%d  logit=%.4f  top5=[%d %d %d %d %d]\n"
        "  argmax match        : %s\n"
        "  top-5 overlap       : %d/5\n"
        "  logit RMSE          : %.4f\n"
        "  logit MAE           : %.4f\n"
        "  baseline logit range: [%.3f, %.3f]\n",
        baseline.argmax_id, baseline.argmax_logit,
        baseline.top5[0], baseline.top5[1], baseline.top5[2], baseline.top5[3], baseline.top5[4],
        bitnet.argmax_id, bitnet.argmax_logit,
        bitnet.top5[0], bitnet.top5[1], bitnet.top5[2], bitnet.top5[3], bitnet.top5[4],
        baseline.argmax_id == bitnet.argmax_id ? "YES" : "NO",
        top5_overlap, rmse, mae,
        *std::min_element(baseline.logits.begin(), baseline.logits.end()),
        *std::max_element(baseline.logits.begin(), baseline.logits.end()));

    // Assertions are lax -- this is exploratory. The point is the printed table.
    CHECK(baseline.argmax_id >= 0);
    CHECK(bitnet.argmax_id >= 0);
}
