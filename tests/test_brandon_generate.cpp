// F5.4g: end-to-end multi-token generation with brandon-tiny on the ternary substrate.
// Sampling recipe per integration guide §6: temp=0.7, top_k=50, top_p=0.9,
// rep_penalty=1.2, no_repeat_ngram_size=3.
//
// Encode a short prompt via SPM, prefill registers, forward each prompt token,
// then sample N continuation tokens. Verify all tokens are valid + decode produces
// a non-empty string. SKIPS gracefully if the GGUF is missing.
//
// Runtime: ~12s/forward × (4 prefill + 1 prompt + 4 gen) ≈ 110s.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/tx/transformer.hpp>
#include <ter/tx/lut_setup.hpp>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <model/loader.h>
#include <inference/tokenizer.h>
#include <inference/sampler.h>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

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

TEST_CASE("brandon-tiny: multi-token generation with brandon sampling recipe") {
    if (!file_exists(GGUF_PATH)) {
        MESSAGE("brandon-tiny GGUF not found -- skipping");
        return;
    }

    nt::GGUFLoader loader;
    REQUIRE(loader.load(GGUF_PATH));
    BrandonTransformer tx = load_brandon_transformer(loader, /*max_seq=*/32);

    // Tokenizer
    nt::Tokenizer tok;
    tok.init(loader.vocab(), loader.config().bos_token_id, loader.config().eos_token_id);

    // Sampler with brandon recipe
    nt::SamplerConfig sc;
    sc.temperature = 0.7f;
    sc.top_k = 50;
    sc.top_p = 0.9f;
    sc.repeat_penalty = 1.2f;
    sc.no_repeat_ngram_size = 3;
    sc.seed = 12345;
    nt::Sampler sampler;
    sampler.init(sc);

    // Sim + kernels + LUTs
    Sim s(64 * 1024);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    LutAddrs luts = load_default_luts(s, "lut_data");

    BrandonState state;
    state.use_value_residual = tx.use_value_residual;
    state.use_dwa            = tx.use_dwa;
    state.n_layers           = tx.n_layers;
    state.hidden_size        = tx.hidden_size;
    if (state.use_dwa) {
        state.dwa_buf.assign(static_cast<size_t>((tx.n_layers + 1) * tx.hidden_size), 0.0f);
        state.dwa_weights = tx.dwa_w.data();
    }

    // 1) Register prefill (positions 0..3).
    int pos = register_prefill(s, kt, tx, luts, &state);
    REQUIRE(pos == tx.n_registers);

    // 2) Encode prompt (raw text, no BOS — per chat.py reference in guide §7).
    auto prompt_ids = tok.encode("hello", /*add_bos=*/false);
    MESSAGE("prompt encodes to ", prompt_ids.size(), " tokens");
    REQUIRE(!prompt_ids.empty());

    // 3) Forward each prompt token. Keep only the LAST forward's logits.
    std::vector<float> last_logits;
    std::vector<int> history;
    for (int t : prompt_ids) {
        last_logits = forward_token(s, kt, tx, t, pos, luts, &state);
        history.push_back(t);
        ++pos;
    }
    REQUIRE(last_logits.size() == static_cast<size_t>(tx.vocab_size));

    // 4) Generate 4 tokens with the brandon sampling recipe.
    constexpr int N_GEN = 4;
    std::vector<int> generated;
    for (int g = 0; g < N_GEN; ++g) {
        std::vector<float> logits_copy = last_logits;
        sampler.apply_repeat_penalty(logits_copy.data(), tx.vocab_size, history);
        sampler.apply_no_repeat_ngram(logits_copy.data(), tx.vocab_size, history);

        int next = sampler.sample(logits_copy.data(), tx.vocab_size);
        REQUIRE(next >= 0);
        REQUIRE(next < tx.vocab_size);
        generated.push_back(next);
        history.push_back(next);

        if (next == tx.vocab_size - 1 || next == tok.eos_id()) break;   // EOS check

        // Forward the new token to get next-step logits.
        last_logits = forward_token(s, kt, tx, next, pos, luts, &state);
        ++pos;
    }

    REQUIRE(generated.size() > 0);

    // Decode generated portion only.
    std::string out_text = tok.decode(generated);
    MESSAGE("generated ", generated.size(), " tokens; ids: ");
    for (int id : generated) MESSAGE("  ", id);
    MESSAGE("decode -> \"", out_text, "\"");

    // Soft sanity: at least we didn't emit only padding/EOS.
    int n_real = 0;
    for (int id : generated) if (id != 0 && id != tok.eos_id()) ++n_real;
    CHECK(n_real > 0);
}
