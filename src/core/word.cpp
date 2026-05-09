#include <ter/word.hpp>

namespace ter {

Word27 Word27::from_int(int64_t v) noexcept {
    Word27 w;
    for (int i = 0; i < kTrits; ++i) {
        int r = static_cast<int>(((v % 3) + 3) % 3);
        int digit = (r == 2) ? -1 : r;
        w.trits_[i] = Trit{digit};
        v = (v - digit) / 3;
    }
    return w;
}

int64_t Word27::to_int() const noexcept {
    int64_t acc = 0;
    int64_t place = 1;
    for (int i = 0; i < kTrits; ++i) {
        acc += static_cast<int64_t>(trits_[i].value()) * place;
        place *= 3;
    }
    return acc;
}

Word27 Word27::operator-() const noexcept {
    Word27 r;
    for (int i = 0; i < kTrits; ++i) r.trits_[i] = -trits_[i];
    return r;
}

Word27 Word27::operator+(Word27 o) const noexcept {
    Word27 r;
    Trit carry = T_ZERO;
    for (int i = 0; i < kTrits; ++i) {
        auto rr = trit_full_add(trits_[i], o.trits_[i], carry);
        r.trits_[i] = rr.sum;
        carry = rr.carry;
    }
    return r;
}

Word27 Word27::operator-(Word27 o) const noexcept {
    return *this + (-o);
}

Trit sign_trit(Word27 w) noexcept {
    for (int i = Word27::kTrits - 1; i >= 0; --i) {
        if (!w.trit(i).is_zero()) return w.trit(i);
    }
    return T_ZERO;
}

}
