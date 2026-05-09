#include <ter/pack.hpp>

namespace ter {

uint64_t pack_word27(const Word27& w) noexcept {
    uint64_t out = 0;
    for (int i = 0; i < Word27::kTrits; ++i) {
        out |= static_cast<uint64_t>(trit_to_code(w.trit(i))) << (i * 2);
    }
    return out;
}

Word27 unpack_word27(uint64_t packed) noexcept {
    Word27 w;
    for (int i = 0; i < Word27::kTrits; ++i) {
        uint8_t code = static_cast<uint8_t>((packed >> (i * 2)) & 0b11);
        w.set_trit(i, code_to_trit(code));
    }
    return w;
}

}
