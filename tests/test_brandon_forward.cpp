// F5.4e smoke test: load brandon-tiny, run one forward_token, verify finite logits + argmax in vocab.
// SKIPS gracefully if the GGUF is missing.
//
// This is the FIRST FULL FORWARD PASS through the ternary substrate on real Llama weights.
// Output is expected to be GARBAGE (skips brandon-specific bits: register prefill, value_residual,
// DenseFormer DWA mixing — F5.4f adds those). Goal here is structural correctness only.
//
// Wall time: ~5-15 seconds for one token (24 layers × ~28k TVMACs/layer = ~670k matmul tiles).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/tx/transformer.hpp>
#include <ter/tx/lut_setup.hpp>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <model/loader.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

#ifndef TER_KERNELS_DIR
#error "TER_KERNELS_DIR must be defined"
#endif

using namespace ter;
using namespace ter::tx;

namespace {
constexpr const char* GGUF_PATH = "/Users/pc/osito-a-models/build/brandon-tiny-10m-f16.gguf";

bool file_exists(const char* path) {
    std::ifstream f(path);
    return f.good();
}
}  // namespace

TEST_CASE("brandon-tiny: load + forward one token through 24 layers") {
    if (!file_exists(GGUF_PATH)) {
        MESSAGE("brandon-tiny GGUF not found -- skipping");
        return;
    }

    nt::GGUFLoader loader;
    REQUIRE(loader.load(GGUF_PATH));

    BrandonTransformer tx = load_brandon_transformer(loader, /*max_seq=*/8);
    MESSAGE("loaded ", tx.blocks.size(), " unique blocks; ",
            tx.n_layers, " logical layers; vocab=", tx.vocab_size);

    REQUIRE(tx.blocks.size() == 12);
    REQUIRE(tx.n_layers == 24);
    REQUIRE(tx.layer_map.size() == 24);
    REQUIRE(tx.vocab_size == 8192);

    Sim s(64 * 1024);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    LutAddrs luts = load_default_luts(s, "lut_data");

    // Forward one token. Pick id 100 — arbitrary, well within vocab.
    int token_id = 100;
    auto logits = forward_token(s, kt, tx, token_id, /*pos=*/0, luts);

    REQUIRE(logits.size() == static_cast<size_t>(tx.vocab_size));

    // Sanity: all logits must be finite (no NaN / Inf).
    int n_finite = 0;
    for (float v : logits) if (std::isfinite(v)) ++n_finite;
    CHECK(n_finite == tx.vocab_size);

    // Sanity: logits aren't all zero (means the layer chain produced nothing).
    double abs_sum = 0.0;
    for (float v : logits) abs_sum += std::fabs(static_cast<double>(v));
    CHECK(abs_sum > 0.0);

    // Argmax must be a valid token id.
    int argmax = static_cast<int>(std::distance(
        logits.begin(),
        std::max_element(logits.begin(), logits.end())));
    CHECK(argmax >= 0);
    CHECK(argmax < tx.vocab_size);

    MESSAGE("first token forward: |logits|_1 = ", abs_sum,
            "  argmax = ", argmax,
            "  TVMAC count = ", s.counters().get(Opcode::TVMAC));
}
