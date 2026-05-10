// F5.4i: TinyStories Llama-2 20M Q4_K_M smoke test.
// Tests the Q4_K_M and Q6_K dequant paths end-to-end (TinyStories mixes both:
// most projections are Q4_K, output.weight and a few ffn_down are Q6_K).
// Loads, runs ONE forward at pos=0, asserts logits finite. Should run in
// seconds (4 layers, hidden=256), unlike the multi-minute Llama 1B smoke.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/tx/transformer.hpp>
#include <ter/tx/lut_setup.hpp>
#include <ter/tx/op_stats.hpp>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <model/loader.h>
#include <fstream>
#include <cmath>
#include <limits>

#ifndef TER_KERNELS_DIR
#error "TER_KERNELS_DIR must be defined"
#endif

namespace {
constexpr const char* GGUF_PATH = "/Users/pc/osito-a-models/build/tinystories-20m-q4km.gguf";

bool file_exists(const char* path) {
    std::ifstream f(path);
    return f.good();
}
}  // namespace

TEST_CASE("TinyStories 20M Q4_K_M: load + 1 forward + finite logits") {
    if (!file_exists(GGUF_PATH)) {
        MESSAGE("TinyStories GGUF not found -- skipping");
        return;
    }
    using namespace ter;
    using namespace ter::tx;

    nt::GGUFLoader loader;
    REQUIRE(loader.load(GGUF_PATH));

    const auto& cfg = loader.config();
    MESSAGE("TS config: hidden=", cfg.hidden_size,
            " ffn=", cfg.intermediate_size,
            " layers=", cfg.n_layers,
            " heads=", cfg.n_heads,
            " kv_heads=", cfg.n_kv_heads,
            " head_dim=", cfg.head_dim);

    BrandonTransformer tx = load_llama_transformer(loader, /*max_seq=*/8, /*n_trits=*/9);

    REQUIRE(tx.n_layers > 0);
    REQUIRE(tx.vocab_size == 32000);
    REQUIRE(tx.hidden_size == 256);
    // TinyStories has separate output.weight, so weight_tying should flip off.
    CHECK_FALSE(tx.weight_tying);
    CHECK(tx.lm_head.payload.size() > 0);

    Sim s(64 * 1024);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    LutAddrs luts = load_default_luts(s, "lut_data");

    int bos = cfg.bos_token_id > 0 ? cfg.bos_token_id : 1;
    if (bos >= tx.vocab_size) bos = 1;

    s.counters().reset();
    auto logits = forward_token(s, kt, tx, bos, /*pos=*/0, luts, /*state=*/nullptr);
    dump_op_stats(s, "TinyStories 20M Q4_K_M, 1 forward",
                  tx.hidden_size, tx.intermediate_size,
                  tx.n_heads, tx.n_kv_heads, tx.head_dim,
                  tx.n_layers, tx.vocab_size);

    REQUIRE(static_cast<int>(logits.size()) == tx.vocab_size);
    int n_finite = 0;
    float lmin =  std::numeric_limits<float>::infinity();
    float lmax = -std::numeric_limits<float>::infinity();
    for (float v : logits) {
        if (std::isfinite(v)) {
            ++n_finite;
            if (v < lmin) lmin = v;
            if (v > lmax) lmax = v;
        }
    }
    MESSAGE("logits: ", n_finite, "/", logits.size(),
            " finite, range [", lmin, ", ", lmax, "]");
    CHECK(n_finite == static_cast<int>(logits.size()));
}
