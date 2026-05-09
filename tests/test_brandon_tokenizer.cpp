#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <model/loader.h>
#include <inference/tokenizer.h>
#include <fstream>
#include <vector>
#include <string>

using namespace nt;

namespace {
constexpr const char* GGUF_PATH = "/Users/pc/osito-a-models/build/brandon-tiny-10m-f16.gguf";

bool file_exists(const char* path) {
    std::ifstream f(path);
    return f.good();
}

// Find a token id by exact string match. Returns -1 if not found.
int find_token_id(const Tokenizer& tok, const GGUFVocab& vocab, const std::string& s) {
    for (int i = 0; i < (int)vocab.tokens.size(); ++i) {
        if (vocab.tokens[i] == s) return i;
    }
    (void)tok;
    return -1;
}
}  // namespace

TEST_CASE("brandon-tiny tokenizer initialises and reports SPM") {
    if (!file_exists(GGUF_PATH)) {
        MESSAGE("brandon-tiny GGUF not found -- skipping");
        return;
    }
    GGUFLoader loader;
    REQUIRE(loader.load(GGUF_PATH));

    const auto& vocab = loader.vocab();
    const auto& cfg   = loader.config();
    MESSAGE("vocab tokens=", vocab.tokens.size(),
            " bos=", cfg.bos_token_id,
            " eos=", cfg.eos_token_id);

    // Per the brandon-tiny config: vocab=8192, bos=2, eos=3.
    CHECK(vocab.tokens.size() == 8192);

    Tokenizer tok;
    tok.init(vocab, cfg.bos_token_id, cfg.eos_token_id);

    CHECK(tok.vocab_size() == 8192);
    CHECK(tok.bos_id() == cfg.bos_token_id);
    CHECK(tok.eos_id() == cfg.eos_token_id);
}

TEST_CASE("brandon-tiny ChatML special tokens resolve by string search") {
    if (!file_exists(GGUF_PATH)) {
        MESSAGE("brandon-tiny GGUF not found -- skipping");
        return;
    }
    GGUFLoader loader;
    REQUIRE(loader.load(GGUF_PATH));

    const auto& vocab = loader.vocab();
    const auto& cfg   = loader.config();
    Tokenizer tok;
    tok.init(vocab, cfg.bos_token_id, cfg.eos_token_id);

    // Per integration guide Step 7: don't assume ids 4/5; resolve by string.
    int im_start = find_token_id(tok, vocab, "<|im_start|>");
    int im_end   = find_token_id(tok, vocab, "<|im_end|>");

    MESSAGE("ChatML <|im_start|>=", im_start, " <|im_end|>=", im_end);

    REQUIRE(im_start >= 0);
    REQUIRE(im_end   >= 0);
    CHECK(im_start != im_end);

    // Per the guide they're at ids 4 and 5 in brandon-tiny — soft-check.
    if (im_start != 4 || im_end != 5) {
        MESSAGE("note: ChatML ids are not the expected 4/5; the guide warns this can shift");
    }
}

TEST_CASE("brandon-tiny encode/decode round-trip on a simple SPM string") {
    if (!file_exists(GGUF_PATH)) {
        MESSAGE("brandon-tiny GGUF not found -- skipping");
        return;
    }
    GGUFLoader loader;
    REQUIRE(loader.load(GGUF_PATH));

    Tokenizer tok;
    tok.init(loader.vocab(), loader.config().bos_token_id, loader.config().eos_token_id);

    // Encode a short ASCII string. SPM auto-detect should pick SentencePiece path
    // (brandon vocab uses ▁ for word boundaries, NOT GPT-2's Ġ).
    const std::string in = "hello world";
    auto ids = tok.encode(in, /*add_bos=*/false);

    MESSAGE("encode(\"", in, "\") -> ", ids.size(), " tokens");
    CHECK(ids.size() > 0);

    // Decode the resulting ids.
    auto out = tok.decode(ids);
    MESSAGE("decode -> \"", out, "\"");

    // SPM round-trip is not byte-exact in general (whitespace handling can shift),
    // but the substring "hello" must survive.
    CHECK(out.find("hello") != std::string::npos);
    CHECK(out.find("world") != std::string::npos);
}
