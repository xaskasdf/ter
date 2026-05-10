// F9.fwd: end-to-end BitNet b1.58 forward on the ternary substrate.
// Loads microsoft/bitnet-b1.58-2B-4T-gguf (i2_s -> {-1,0,+1} via dequant_i2_s),
// runs ONE forward at pos=0 from BOS, asserts finite logits over vocab 128256.
//
// Substrate-data alignment: weights are natively ternary; the existing
// tk_matmul_b_9t kernel computes correct results because Format B with
// {-1,0,+1} payload values is just a degenerate case of the integer matmul.
// The op count still shows non-zero TVMAC because we use the generic
// kernel; a dedicated tk_matmul_bitnet (TVADD/TVSUB only) would zero them.
//
// Runtime: 30 layers x 7 matmuls each at hidden=2560/ffn=6912 -- expect
// ~15-20 min per forward. Skips cleanly if the GGUF is absent.
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
constexpr const char* GGUF_PATH =
    "/Users/pc/osito-a-models/downloads/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf";

bool file_exists(const char* path) {
    std::ifstream f(path);
    return f.good();
}
}  // namespace

TEST_CASE("BitNet b1.58 2B-4T: load + 1 forward + finite logits") {
    if (!file_exists(GGUF_PATH)) {
        MESSAGE("BitNet GGUF not found -- skipping");
        return;
    }
    using namespace ter;
    using namespace ter::tx;

    nt::GGUFLoader loader;
    REQUIRE(loader.load(GGUF_PATH));

    const auto& cfg = loader.config();
    MESSAGE("BitNet config: arch=", cfg.architecture,
            " hidden=", cfg.hidden_size,
            " ffn=", cfg.intermediate_size,
            " layers=", cfg.n_layers,
            " heads=", cfg.n_heads,
            " kv_heads=", cfg.n_kv_heads,
            " head_dim=", cfg.head_dim);

    BrandonTransformer tx = load_bitnet_transformer(loader, /*max_seq=*/8, /*n_trits=*/9);

    REQUIRE(tx.n_layers == 30);
    REQUIRE(tx.hidden_size == 2560);
    REQUIRE(tx.vocab_size == 128256);
    // Sub-norms are present.
    REQUIRE(tx.blocks.size() == 30);
    CHECK(tx.blocks[0].attn_sub_norm_w.size() == 2560);
    CHECK(tx.blocks[0].ffn_sub_norm_w.size() == 6912);

    Sim s(64 * 1024);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    LutAddrs luts = load_default_luts(s, "lut_data");

    int bos = cfg.bos_token_id > 0 ? cfg.bos_token_id : 128000;
    if (bos >= tx.vocab_size) bos = 128000;
    MESSAGE("Forward at pos=0 with token id=", bos);

    s.counters().reset();
    auto logits = forward_token(s, kt, tx, bos, /*pos=*/0, luts, /*state=*/nullptr);
    dump_op_stats(s, "BitNet b1.58 2B, 1 forward @ pos=0",
                  tx.hidden_size, tx.intermediate_size,
                  tx.n_heads, tx.n_kv_heads, tx.head_dim,
                  tx.n_layers, tx.vocab_size, /*n_forwards=*/1);

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
