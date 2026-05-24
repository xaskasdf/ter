#include <tersim/bp_model.hpp>
#include <cassert>

namespace tersim {

BranchPredictor::BranchPredictor(BpConfig cfg)
  : cfg_(cfg),
    pht_(static_cast<std::size_t>(cfg.pht_entries), 0b01),   // weak not-taken
    btb_target_(static_cast<std::size_t>(cfg.btb_entries), 0),
    btb_valid_(static_cast<std::size_t>(cfg.btb_entries), false) {
    assert(cfg.pht_entries > 0 && cfg.btb_entries > 0);
    assert(cfg.ghr_history_bits >= 0 && cfg.ghr_history_bits < 64);
    ghr_mask_ = (cfg.ghr_history_bits == 0)
                    ? 0
                    : ((std::uint64_t{1} << cfg.ghr_history_bits) - 1);
}

std::size_t BranchPredictor::pht_index(std::uint64_t pc) const noexcept {
    // gshare: PC xor GHR, masked to PHT size. PC already byte-aligned in trit
    // ISA (one Word27 per instr); we shift by 0 since trit-word indexing is 1c.
    const auto xored = pc ^ ghr_;
    return static_cast<std::size_t>(xored % static_cast<std::uint64_t>(cfg_.pht_entries));
}

std::size_t BranchPredictor::btb_index(std::uint64_t pc) const noexcept {
    return static_cast<std::size_t>(pc % static_cast<std::uint64_t>(cfg_.btb_entries));
}

bool BranchPredictor::predict_and_update(std::uint64_t pc, bool taken,
                                          std::uint64_t target) noexcept {
    ++stats_.branches;

    bool correct;
    if (cfg_.perfect) {
        correct = true;
    } else {
        const auto idx_pht = pht_index(pc);
        const auto cnt     = pht_[idx_pht];
        const bool pred_taken = (cnt >= 2);

        // Direction prediction correctness.
        correct = (pred_taken == taken);

        // BTB target check on a predicted-taken outcome that's actually taken:
        // if BTB has no entry or wrong target, treat as mispredict.
        if (correct && taken) {
            const auto idx_btb = btb_index(pc);
            if (!btb_valid_[idx_btb] || btb_target_[idx_btb] != target) {
                correct = false;
            }
        }

        // Update saturating 2-bit counter.
        if (taken && cnt < 3) pht_[idx_pht] = static_cast<std::uint8_t>(cnt + 1);
        if (!taken && cnt > 0) pht_[idx_pht] = static_cast<std::uint8_t>(cnt - 1);

        // Update BTB on any taken branch (regardless of prediction).
        if (taken) {
            const auto idx_btb = btb_index(pc);
            btb_target_[idx_btb] = target;
            btb_valid_[idx_btb]  = true;
        }

        // Update GHR with actual outcome.
        ghr_ = ((ghr_ << 1) | (taken ? 1u : 0u)) & ghr_mask_;
    }

    if (correct) ++stats_.correct;
    else         ++stats_.mispredicts;
    return correct;
}

}  // namespace tersim
