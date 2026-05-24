#pragma once
#include <cstdint>
#include <vector>

namespace tersim {

struct BpConfig {
    int  pht_entries        = 4096;   // gshare PHT, 2-bit counters
    int  ghr_history_bits   = 12;     // log2(pht_entries) by default
    int  btb_entries        = 1024;   // direct-mapped target buffer
    bool perfect            = false;  // true → always correct (roofline mode)
    int  mispredict_penalty = 4;      // squashed wrong-path stages (IF..EX-1)
};

struct BpStats {
    std::uint64_t branches    = 0;
    std::uint64_t correct     = 0;
    std::uint64_t mispredicts = 0;
    double accuracy() const noexcept {
        return branches ? static_cast<double>(correct) / branches : 0.0;
    }
};

// gshare direction predictor + tiny direct-mapped BTB. We only track direction
// mispredicts for now (target mispredicts on BTB miss are folded into the
// direction penalty for simplicity).
class BranchPredictor {
public:
    explicit BranchPredictor(BpConfig cfg = {});

    // Predict + update with ground truth. Returns true if the prediction was
    // correct (no penalty); false → caller applies mispredict_penalty().
    bool predict_and_update(std::uint64_t pc, bool taken, std::uint64_t target) noexcept;

    int  mispredict_penalty() const noexcept { return cfg_.perfect ? 0 : cfg_.mispredict_penalty; }
    const BpStats& stats()   const noexcept { return stats_; }
    const BpConfig& config() const noexcept { return cfg_; }

private:
    BpConfig                cfg_;
    std::vector<std::uint8_t>  pht_;        // 2-bit saturating counters (0..3)
    std::vector<std::uint64_t> btb_target_;
    std::vector<bool>          btb_valid_;
    std::uint64_t              ghr_     = 0;  // lower ghr_history_bits bits used
    std::uint64_t              ghr_mask_ = 0;
    BpStats                    stats_;

    std::size_t pht_index(std::uint64_t pc) const noexcept;
    std::size_t btb_index(std::uint64_t pc) const noexcept;
};

}  // namespace tersim
