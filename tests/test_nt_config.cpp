#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <model/config.h>

using namespace nt;

TEST_CASE("ModelConfig defaults are sensible") {
    ModelConfig c;
    CHECK(c.vocab_size == 32000);
    CHECK(c.hidden_size == 4096);
    CHECK(c.n_heads == 32);
    CHECK(c.is_gqa() == false);
    CHECK(c.group_size() == 1);
}

TEST_CASE("GQA detection works") {
    ModelConfig c;
    c.n_heads = 32;
    c.n_kv_heads = 8;
    CHECK(c.is_gqa());
    CHECK(c.group_size() == 4);
}
