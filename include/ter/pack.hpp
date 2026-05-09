#pragma once
#include <ter/trit.hpp>
#include <ter/word.hpp>
#include <cstdint>

namespace ter {

constexpr uint8_t trit_to_code(Trit t) noexcept {
    int v = t.value();
    return v == 0 ? 0b00 : v == +1 ? 0b01 : 0b10;
}

constexpr Trit code_to_trit(uint8_t c) noexcept {
    switch (c & 0b11) {
        case 0b00: return T_ZERO;
        case 0b01: return T_POS;
        case 0b10: return T_NEG;
        default:   return T_ZERO;
    }
}

uint64_t pack_word27(const Word27& w) noexcept;
Word27   unpack_word27(uint64_t packed) noexcept;

}
