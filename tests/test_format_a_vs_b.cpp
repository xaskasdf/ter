// F7.x: Format A vs Format B encoding-precision comparison on TinyStories.
// Runs the SAME forward twice -- once with weights quantized straight to
// Format B (9 trits/elem), once with a TFloat (Format A: 1+5+9 = 15 trits/elem)
// round-trip applied to the dequantized float buffer before Format B
// quantization. Reports argmax delta and logit RMSE.
//
// Hypothesis (spec §10 H2): the wider Format A representation should preserve
// quality (matching argmax, low logit RMSE) -- demonstrating the radix-3
// information-density advantage in the encoding step.
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
#include <cstdio>

#ifndef TER_KERNELS_DIR
#error "TER_KERNELS_DIR must be defined"
#endif

namespace {
constexpr const char* GGUF_PATH = "/Users/pc/osito-a-models/build/tinystories-20m-q4km.gguf";

bool file_exists(const char* path) {
    std::ifstream f(path);
    return f.good();
}

struct Run {
    int argmax_id = -1;
    float argmax_logit = 0.0f;
    std::vector<float> logits;
};

Run run_one(bool format_a, int bos) {
    using namespace ter;
    using namespace ter::tx;

    nt::GGUFLoader loader;
    REQUIRE(loader.load(GGUF_PATH));
    BrandonTransformer tx = load_llama_transformer(loader, /*max_seq=*/8,
                                                   /*n_trits=*/9, format_a);

    Sim s(64 * 1024);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    LutAddrs luts = load_default_luts(s, "lut_data");

    Run r;
    r.logits = forward_token(s, kt, tx, bos, /*pos=*/0, luts, /*state=*/nullptr);
    r.argmax_id = nt::Sampler::argmax(r.logits.data(), tx.vocab_size);
    r.argmax_logit = r.logits[static_cast<size_t>(r.argmax_id)];
    return r;
}
}  // namespace

TEST_CASE("Format A vs B: TinyStories argmax + RMSE comparison") {
    if (!file_exists(GGUF_PATH)) {
        MESSAGE("TinyStories GGUF not found -- skipping");
        return;
    }

    int bos = 1;  // TinyStories tokenizer BOS

    auto rb = run_one(/*format_a=*/false, bos);
    auto ra = run_one(/*format_a=*/true,  bos);

    REQUIRE(rb.logits.size() == ra.logits.size());

    double sum_sq = 0.0;
    double sum_abs = 0.0;
    for (size_t i = 0; i < rb.logits.size(); ++i) {
        double d = static_cast<double>(rb.logits[i]) - static_cast<double>(ra.logits[i]);
        sum_sq  += d * d;
        sum_abs += std::fabs(d);
    }
    double rmse = std::sqrt(sum_sq / static_cast<double>(rb.logits.size()));
    double mae  = sum_abs / static_cast<double>(rb.logits.size());

    std::fprintf(stderr,
        "\n=== Format A vs B (TinyStories, BOS=%d) ===\n"
        "  Format B argmax = %d (logit %.4f)\n"
        "  Format A argmax = %d (logit %.4f)\n"
        "  argmax match    = %s\n"
        "  logit RMSE      = %.6f\n"
        "  logit MAE       = %.6f\n"
        "  trit cost       = B:9 trits/elem  vs  A:15 trits/elem (1+5+9)\n",
        bos, rb.argmax_id, rb.argmax_logit, ra.argmax_id, ra.argmax_logit,
        rb.argmax_id == ra.argmax_id ? "YES" : "NO",
        rmse, mae);

    // Format A is wider (15 trits) and should preserve at minimum the same
    // top token as Format B (9 trits). Tight RMSE bound is the real claim.
    CHECK(rb.argmax_id == ra.argmax_id);
    CHECK(rmse < 0.5);   // logit-space delta should be small
}
