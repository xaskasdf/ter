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

Word54 Word54::from_int(int64_t v) noexcept {
    Word54 w;
    for (int i = 0; i < kTrits; ++i) {
        int r = static_cast<int>(((v % 3) + 3) % 3);
        int digit = (r == 2) ? -1 : r;
        w.trits_[i] = Trit{digit};
        v = (v - digit) / 3;
        if (v == 0) break;
    }
    return w;
}

int64_t Word54::to_int() const noexcept {
    __int128 acc = 0, place = 1;
    for (int i = 0; i < kTrits; ++i) {
        acc += static_cast<__int128>(trits_[i].value()) * place;
        place *= 3;
    }
    return static_cast<int64_t>(acc);
}

Word54 Word54::operator+(Word54 o) const noexcept {
    Word54 r;
    Trit carry = T_ZERO;
    for (int i = 0; i < kTrits; ++i) {
        auto rr = trit_full_add(trits_[i], o.trits_[i], carry);
        r.trits_[i] = rr.sum;
        carry = rr.carry;
    }
    return r;
}

Word54 Word54::operator-() const noexcept {
    Word54 r;
    for (int i = 0; i < kTrits; ++i) r.trits_[i] = -trits_[i];
    return r;
}

Word54 mul(Word27 a, Word27 b) noexcept {
    Word54 acc;
    for (int j = 0; j < Word27::kTrits; ++j) {
        Trit bj = b.trit(j);
        if (bj.is_zero()) continue;
        Word54 partial;
        for (int i = 0; i < Word27::kTrits; ++i) {
            int dst = i + j;
            if (dst >= Word54::kTrits) break;
            Trit ai = a.trit(i);
            partial.set_trit(dst, bj.value() < 0 ? -ai : ai);
        }
        acc = acc + partial;
    }
    return acc;
}

void mac_inplace(Word54& acc, Word27 a, Word27 b) noexcept {
    acc = acc + mul(a, b);
}

}
