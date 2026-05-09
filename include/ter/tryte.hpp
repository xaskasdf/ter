#pragma once
#include <ter/trit.hpp>
#include <array>
#include <cassert>

namespace ter {

class Tryte {
public:
    constexpr Tryte() noexcept = default;

    static constexpr int min_int() noexcept { return -13; }
    static constexpr int max_int() noexcept { return +13; }

    static constexpr Tryte from_int(int v) noexcept {
        assert(v >= min_int() && v <= max_int());
        Tryte t;
        for (int i = 0; i < 3; ++i) {
            int r = ((v % 3) + 3) % 3;
            int digit = (r == 2) ? -1 : r;
            t.trits_[i] = Trit{digit};
            v = (v - digit) / 3;
        }
        return t;
    }

    constexpr int to_int() const noexcept {
        int acc = 0, place = 1;
        for (int i = 0; i < 3; ++i) {
            acc += trits_[i].value() * place;
            place *= 3;
        }
        return acc;
    }

    constexpr Trit trit(int i) const noexcept { return trits_[i]; }

    constexpr Tryte operator-() const noexcept {
        Tryte r;
        for (int i = 0; i < 3; ++i) r.trits_[i] = -trits_[i];
        return r;
    }

    constexpr bool operator==(Tryte o) const noexcept {
        for (int i = 0; i < 3; ++i) if (trits_[i] != o.trits_[i]) return false;
        return true;
    }
    constexpr bool operator!=(Tryte o) const noexcept { return !(*this == o); }

private:
    std::array<Trit, 3> trits_{};
};

}
