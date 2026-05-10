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

Run run_one(bool format_a, int mant_trits, int bos) {
    using namespace ter;
    using namespace ter::tx;

    nt::GGUFLoader loader;
    REQUIRE(loader.load(GGUF_PATH));
    BrandonTransformer tx = load_llama_transformer(loader, /*max_seq=*/8,
                                                   /*n_trits=*/9, format_a, mant_trits);

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

    auto rb    = run_one(/*format_a=*/false, /*mant_trits=*/9, bos);  // baseline
    auto ra15  = run_one(/*format_a=*/true,  /*mant_trits=*/9, bos);  // 1+5+9 = 15 trits
    auto ra11  = run_one(/*format_a=*/true,  /*mant_trits=*/5, bos);  // 1+5+5 = 11 trits
    auto ra7   = run_one(/*format_a=*/true,  /*mant_trits=*/1, bos);  // 1+5+1 = 7 trits (extreme)

    REQUIRE(rb.logits.size() == ra15.logits.size());

    auto rmse = [&](const Run& a, const Run& b) {
        double s = 0.0;
        for (size_t i = 0; i < a.logits.size(); ++i) {
            double d = a.logits[i] - b.logits[i];
            s += d * d;
        }
        return std::sqrt(s / static_cast<double>(a.logits.size()));
    };

    std::fprintf(stderr,
        "\n=== Format A trit-budget sweep vs Format B baseline (TinyStories) ===\n"
        "                       argmax  top-logit  vs-B-RMSE   trits/elem\n"
        "  Format B 9-trit    : %6d  %9.4f  --------    9\n"
        "  Format A 15-trit   : %6d  %9.4f  %.6f   15  (1+5+9)\n"
        "  Format A 11-trit   : %6d  %9.4f  %.6f   11  (1+5+5)\n"
        "  Format A  7-trit   : %6d  %9.4f  %.6f    7  (1+5+1)\n",
        rb.argmax_id, rb.argmax_logit,
        ra15.argmax_id, ra15.argmax_logit, rmse(rb, ra15),
        ra11.argmax_id, ra11.argmax_logit, rmse(rb, ra11),
        ra7.argmax_id,  ra7.argmax_logit,  rmse(rb, ra7));

    // 15-trit A is much wider than 9-trit B; should match argmax exactly.
    CHECK(ra15.argmax_id == rb.argmax_id);
    CHECK(rmse(rb, ra15) < 0.01);

    // 11-trit A: H2 hypothesis says fewer total trits than 9-trit B (it's MORE
    // trits, but the win point is the encoded dynamic range, not the count).
    // Under TinyStories' modest precision needs we expect argmax preservation.
    CHECK(ra11.argmax_id == rb.argmax_id);

    // 7-trit A: stress test. Argmax MAY drift; just check the run completes.
    CHECK(ra7.argmax_id >= 0);
}
