// F5.4h: A/B calibration of n_trits per element.
// Greedy decoding (temp=0) for deterministic argmax comparison.
// Generates 2 tokens at each setting and reports argmax + decoded text.
//
// Runtime: 2 configs × (4 prefill + 1 prompt + 2 gen = 7 forwards × ~12s) ≈ 170s.
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

struct GenResult {
    std::vector<int> ids;
    std::string text;
};

// Run greedy generation: 4 register prefills, encode "hello", forward, generate N tokens
// at temperature 0 (argmax). Returns tokens + decode.
GenResult run_greedy(int n_trits, int n_gen) {
    nt::GGUFLoader loader;
    REQUIRE(loader.load(GGUF_PATH));

    BrandonTransformer tx = load_brandon_transformer(loader, /*max_seq=*/16, n_trits);

    nt::Tokenizer tok;
    tok.init(loader.vocab(), loader.config().bos_token_id, loader.config().eos_token_id);

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

    int pos = register_prefill(s, kt, tx, luts, &state);
    auto prompt_ids = tok.encode("hello", /*add_bos=*/false);

    std::vector<float> last_logits;
    std::vector<int> history(prompt_ids.begin(), prompt_ids.end());
    for (int t : prompt_ids) {
        last_logits = forward_token(s, kt, tx, t, pos, luts, &state);
        ++pos;
    }

    GenResult r;
    for (int g = 0; g < n_gen; ++g) {
        int next = nt::Sampler::argmax(last_logits.data(), tx.vocab_size);
        r.ids.push_back(next);
        history.push_back(next);
        if (next == tok.eos_id() || next == 0) break;
        last_logits = forward_token(s, kt, tx, next, pos, luts, &state);
        ++pos;
    }
    r.text = tok.decode(r.ids);
    return r;
}
}  // namespace

TEST_CASE("brandon-tiny calibration: 9 trits vs 12 trits, greedy") {
    if (!file_exists(GGUF_PATH)) {
        MESSAGE("brandon-tiny GGUF not found -- skipping");
        return;
    }

    constexpr int N_GEN = 2;

    auto r9  = run_greedy(/*n_trits=*/9,  N_GEN);
    auto r12 = run_greedy(/*n_trits=*/12, N_GEN);

    MESSAGE("=== 9-trit greedy (baseline) ===");
    for (int id : r9.ids) MESSAGE("  id=", id);
    MESSAGE("decode -> \"", r9.text, "\"");

    MESSAGE("=== 12-trit greedy (calibrated) ===");
    for (int id : r12.ids) MESSAGE("  id=", id);
    MESSAGE("decode -> \"", r12.text, "\"");

    // Both should produce valid token ids.
    CHECK(r9.ids.size()  > 0);
    CHECK(r12.ids.size() > 0);
    for (int id : r9.ids)  { CHECK(id >= 0); CHECK(id < 8192); }
    for (int id : r12.ids) { CHECK(id >= 0); CHECK(id < 8192); }
}
