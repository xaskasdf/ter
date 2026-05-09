// F6.3: Llama 3.2 1B Q8_0 smoke test on the ternary substrate.
// Loads the GGUF, runs ONE forward pass at pos=0 with a single token (BOS),
// asserts the resulting logits vector is the right size and all-finite.
//
// Runtime: per-forward cost dominated by 16 layers * 7 matmuls each at the
// 2048/8192 dimensions; expect several minutes. We skip the test cleanly if
// the model file isn't on disk so the suite stays usable on machines without
// the 1.32 GB Q8_0 download.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/tx/transformer.hpp>
#include <ter/tx/lut_setup.hpp>
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
    "/Users/pc/osito-a-models/downloads/llama-3.2-1b-instruct/llama-3.2-1b-instruct-q8_0.gguf";

bool file_exists(const char* path) {
    std::ifstream f(path);
    return f.good();
}
}  // namespace

TEST_CASE("Llama 3.2 1B Q8_0: load + 1 forward + finite logits") {
    if (!file_exists(GGUF_PATH)) {
        MESSAGE("Llama 3.2 1B GGUF not found at ", GGUF_PATH, " -- skipping");
        return;
    }

    using namespace ter;
    using namespace ter::tx;

    nt::GGUFLoader loader;
    REQUIRE(loader.load(GGUF_PATH));

    const auto& cfg = loader.config();
    MESSAGE("Llama config: hidden=", cfg.hidden_size,
            " ffn=", cfg.intermediate_size,
            " layers=", cfg.n_layers,
            " heads=", cfg.n_heads,
            " kv_heads=", cfg.n_kv_heads,
            " head_dim=", cfg.head_dim);

    BrandonTransformer tx = load_llama_transformer(loader, /*max_seq=*/8, /*n_trits=*/9);

    REQUIRE(tx.n_layers > 0);
    REQUIRE(tx.vocab_size == 128256);
    REQUIRE(tx.hidden_size == 2048);
    REQUIRE(static_cast<int>(tx.blocks.size()) == tx.n_layers);
    REQUIRE(tx.layer_map.size() == static_cast<size_t>(tx.n_layers));
    for (int i = 0; i < tx.n_layers; ++i) CHECK(tx.layer_map[static_cast<size_t>(i)] == i);

    Sim s(64 * 1024);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    LutAddrs luts = load_default_luts(s, "lut_data");

    // BOS token (Llama 3 uses 128000 = <|begin_of_text|>); fall back to 1 if metadata absent.
    int bos = cfg.bos_token_id > 0 ? cfg.bos_token_id : 1;
    if (bos >= tx.vocab_size) bos = 1;
    MESSAGE("Forward at pos=0 with token id=", bos);

    auto logits = forward_token(s, kt, tx, bos, /*pos=*/0, luts, /*state=*/nullptr);

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
