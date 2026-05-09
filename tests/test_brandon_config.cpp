#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <model/loader.h>
#include <fstream>

using namespace nt;

namespace {
constexpr const char* GGUF_PATH = "/Users/pc/osito-a-models/build/brandon-tiny-10m-f16.gguf";

bool file_exists(const char* path) {
    std::ifstream f(path);
    return f.good();
}
}  // namespace

TEST_CASE("brandon-tiny config populates BrandonConfig correctly") {
    if (!file_exists(GGUF_PATH)) {
        MESSAGE("brandon-tiny GGUF not found -- skipping");
        return;
    }

    GGUFLoader loader;
    REQUIRE(loader.load(GGUF_PATH));

    const auto& cfg = loader.config();

    // Standard dims (now correctly populated from brandon.* keys).
    CHECK(cfg.hidden_size       == 256);
    CHECK(cfg.intermediate_size == 720);
    CHECK(cfg.n_heads           == 8);
    CHECK(cfg.n_kv_heads        == 2);
    CHECK(cfg.head_dim          == 32);
    CHECK(cfg.max_seq_len       == 512);
    CHECK(cfg.norm_eps          == doctest::Approx(1e-5f).epsilon(1e-3));
    CHECK(cfg.rope_theta        == doctest::Approx(10000.0f));

    // Brandon-specific.
    const auto& b = cfg.brandon;
    CHECK(b.block_count         == 12);
    CHECK(b.compute_layer_count == 24);
    CHECK(b.n_registers         == 4);
    CHECK(b.n_loops             == 1);
    CHECK(b.use_dwa             == true);
    CHECK(b.use_value_residual  == true);
    CHECK(b.weight_tying        == true);

    REQUIRE(b.layer_map.size() == 24);
    for (auto idx : b.layer_map) {
        CHECK(idx >= 0);
        CHECK(idx < b.block_count);
    }

    // n_layers should be compute_layer_count for forward-loop iteration.
    CHECK(cfg.n_layers == 24);

    CHECK(b.is_valid());
}
