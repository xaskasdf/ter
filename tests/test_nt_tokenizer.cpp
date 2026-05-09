#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <inference/tokenizer.h>
#include <model/loader.h>

using namespace nt;

// Build a minimal GGUFVocab with a few SentencePiece-style tokens
// so Tokenizer::init() can run fully without crashing.
static GGUFVocab make_minimal_vocab() {
    GGUFVocab v;
    // index 0: unknown
    v.tokens.push_back("<unk>");
    v.scores.push_back(0.0f);
    v.token_types.push_back(1);  // unknown

    // index 1: BOS
    v.tokens.push_back("<s>");
    v.scores.push_back(0.0f);
    v.token_types.push_back(2);  // control

    // index 2: EOS
    v.tokens.push_back("</s>");
    v.scores.push_back(0.0f);
    v.token_types.push_back(2);  // control

    // index 3: a normal token (SentencePiece style)
    v.tokens.push_back("\xe2\x96\x81test");  // ▁test
    v.scores.push_back(-1.0f);
    v.token_types.push_back(0);  // normal

    return v;
}

TEST_CASE("Tokenizer initialises and reports bos/eos") {
    GGUFVocab vocab = make_minimal_vocab();
    Tokenizer t;
    t.init(vocab, /*bos*/1, /*eos*/2);
    CHECK(t.bos_id() == 1);
    CHECK(t.eos_id() == 2);
}

TEST_CASE("Tokenizer vocab_size matches input") {
    GGUFVocab vocab = make_minimal_vocab();
    Tokenizer t;
    t.init(vocab, 1, 2);
    CHECK(t.vocab_size() == 4);
}

TEST_CASE("Tokenizer encode empty string returns only BOS") {
    GGUFVocab vocab = make_minimal_vocab();
    Tokenizer t;
    t.init(vocab, 1, 2);
    auto ids = t.encode("", /*add_bos=*/true);
    REQUIRE(ids.size() == 1);
    CHECK(ids[0] == 1);
}

TEST_CASE("Tokenizer decode_token out-of-range returns empty") {
    GGUFVocab vocab = make_minimal_vocab();
    Tokenizer t;
    t.init(vocab, 1, 2);
    CHECK(t.decode_token(-1) == "");
    CHECK(t.decode_token(9999) == "");
}
