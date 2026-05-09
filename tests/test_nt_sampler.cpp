#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <inference/sampler.h>
#include <vector>

using namespace nt;

TEST_CASE("Sampler with temperature 0 returns argmax") {
    Sampler s;
    SamplerConfig cfg;
    cfg.temperature = 0.0f;
    cfg.seed = 42;
    s.init(cfg);
    std::vector<float> logits = {0.1f, 0.5f, 0.9f, 0.3f};
    int tok = s.sample(logits.data(), static_cast<int>(logits.size()));
    CHECK(tok == 2);
}
