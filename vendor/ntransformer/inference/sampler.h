#pragma once

#include <vector>
#include <random>

namespace nt {

// ============================================================
// Token sampler: top-k, top-p (nucleus), temperature
// ============================================================

struct SamplerConfig {
    float temperature = 0.7f;
    int   top_k       = 40;
    float top_p       = 0.9f;
    float repeat_penalty = 1.1f;
    int   repeat_window  = 64;
    int   no_repeat_ngram_size = 0;   // 0 = disabled. Brandon recipe uses 3.
    uint64_t seed     = 42;
};

class Sampler {
public:
    Sampler() = default;

    void init(const SamplerConfig& config);

    // Sample a token from logits [vocab_size] (on CPU)
    int sample(const float* logits, int vocab_size);

    // Apply repeat penalty based on recent tokens
    void apply_repeat_penalty(float* logits, int vocab_size,
                              const std::vector<int>& recent_tokens);

    // Apply no-repeat-ngram ban: if config_.no_repeat_ngram_size = n, scan history
    // for any prior occurrence of the last (n-1) tokens; force logits[history[i+n-1]] = -inf
    // to forbid closing any n-gram already seen. Defends against degenerate loops that
    // repetition_penalty alone cannot break (per brandon integration guide §6).
    void apply_no_repeat_ngram(float* logits, int vocab_size,
                               const std::vector<int>& history);

    // Greedy (argmax)
    static int argmax(const float* logits, int vocab_size);

    // Reset RNG
    void set_seed(uint64_t seed);

private:
    SamplerConfig config_;
    std::mt19937 rng_;

    // Working buffers
    std::vector<std::pair<float, int>> candidates_;
};

} // namespace nt
