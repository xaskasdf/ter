// F6.5: Llama 3.2 1B Q8_0 multi-token greedy generation on the ternary substrate.
// Starts from BOS, runs N_GEN forward passes, picks argmax each step, advances
// pos and KV cache automatically via forward_token. Emits the token id stream.
//
// Runtime: ~8.5 min per forward at the 2048/8192 dimensions; with N_GEN=2 the
// test takes ~25 min. Skips cleanly if the GGUF is absent.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/tx/transformer.hpp>
#include <ter/tx/lut_setup.hpp>
#include <ter/tx/op_stats.hpp>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <model/loader.h>
#include <inference/sampler.h>
#include <fstream>
#include <cstdio>

#ifndef TER_KERNELS_DIR
#error "TER_KERNELS_DIR must be defined"
#endif

namespace {
constexpr const char* GGUF_PATH =
    "/Users/pc/osito-a-models/downloads/llama-3.2-1b-instruct/llama-3.2-1b-instruct-q8_0.gguf";
constexpr int N_GEN = 2;

bool file_exists(const char* path) {
    std::ifstream f(path);
    return f.good();
}
}  // namespace

TEST_CASE("Llama 3.2 1B: greedy multi-token generation from BOS") {
    if (!file_exists(GGUF_PATH)) {
        MESSAGE("Llama 3.2 1B GGUF not found -- skipping");
        return;
    }
    using namespace ter;
    using namespace ter::tx;

    nt::GGUFLoader loader;
    REQUIRE(loader.load(GGUF_PATH));
    BrandonTransformer tx = load_llama_transformer(loader, /*max_seq=*/16, /*n_trits=*/9);

    Sim s(64 * 1024);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    LutAddrs luts = load_default_luts(s, "lut_data");

    int bos = loader.config().bos_token_id > 0 ? loader.config().bos_token_id : 128000;
    if (bos >= tx.vocab_size) bos = 128000;

    std::fprintf(stderr, "\n=== Llama 3.2 1B greedy generation: BOS=%d, N_GEN=%d ===\n",
                 bos, N_GEN);

    int pos = 0;
    int cur = bos;
    s.counters().reset();
    for (int g = 0; g < 1 + N_GEN; ++g) {
        auto logits = forward_token(s, kt, tx, cur, pos, luts, /*state=*/nullptr);
        REQUIRE(static_cast<int>(logits.size()) == tx.vocab_size);
        int next = nt::Sampler::argmax(logits.data(), tx.vocab_size);
        std::fprintf(stderr, "  pos=%d  cur=%d  -> next=%d  (logit=%.3f)\n",
                     pos, cur, next, logits[static_cast<size_t>(next)]);
        CHECK(next >= 0);
        CHECK(next < tx.vocab_size);
        cur = next;
        ++pos;
    }
    dump_op_stats(s, "Llama 3.2 1B, 1 + N_GEN forwards",
                  tx.hidden_size, tx.intermediate_size,
                  tx.n_heads, tx.n_kv_heads, tx.head_dim,
                  tx.n_layers, tx.vocab_size,
                  /*n_forwards=*/1 + N_GEN);
}
