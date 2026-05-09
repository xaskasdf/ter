#pragma once
#include <ter/isa.hpp>
#include <unordered_map>
#include <cstdint>

namespace ter {

class OpCounters {
public:
    void bump(Opcode op) noexcept { ++counts_[static_cast<int16_t>(op)]; ++total_; }
    uint64_t get(Opcode op) const noexcept {
        auto it = counts_.find(static_cast<int16_t>(op));
        return it == counts_.end() ? 0 : it->second;
    }
    uint64_t total() const noexcept { return total_; }
    void reset() noexcept { counts_.clear(); total_ = 0; }

    const std::unordered_map<int16_t, uint64_t>& raw() const noexcept { return counts_; }

private:
    std::unordered_map<int16_t, uint64_t> counts_;
    uint64_t total_ = 0;
};

}
