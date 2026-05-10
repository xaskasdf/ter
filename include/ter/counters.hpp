#pragma once
#include <ter/isa.hpp>
#include <array>
#include <cstdint>
#include <cstring>

namespace ter {

// Flat-array counter table (no std::unordered_map). Opcodes are <256 by
// construction (see isa.hpp; max value used is 122 = TVMUL). Sized at 256
// for headroom and to keep one libstdc++ dependency out of the K4 build.
class OpCounters {
public:
    static constexpr size_t kSlots = 256;

    void bump(Opcode op) noexcept {
        ++counts_[static_cast<size_t>(static_cast<int16_t>(op)) & (kSlots - 1)];
        ++total_;
    }
    void bump_n(Opcode op, uint64_t n) noexcept {
        counts_[static_cast<size_t>(static_cast<int16_t>(op)) & (kSlots - 1)] += n;
        total_ += n;
    }
    uint64_t get(Opcode op) const noexcept {
        return counts_[static_cast<size_t>(static_cast<int16_t>(op)) & (kSlots - 1)];
    }
    uint64_t total() const noexcept { return total_; }
    void reset() noexcept {
        std::memset(counts_.data(), 0, sizeof(uint64_t) * kSlots);
        total_ = 0;
    }

    const std::array<uint64_t, kSlots>& raw() const noexcept { return counts_; }

private:
    std::array<uint64_t, kSlots> counts_{};
    uint64_t total_ = 0;
};

}
