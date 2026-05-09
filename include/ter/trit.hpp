#pragma once
#include <cstdint>

namespace ter {

class Trit {
public:
    constexpr Trit() noexcept : v_(0) {}
    constexpr explicit Trit(int v) noexcept
        : v_(static_cast<int8_t>(v < 0 ? -1 : v > 0 ? 1 : 0)) {}

    constexpr int8_t value() const noexcept { return v_; }
    constexpr bool   is_zero() const noexcept { return v_ == 0; }

    constexpr Trit operator-() const noexcept { return Trit{-v_}; }
    constexpr bool operator==(Trit o) const noexcept { return v_ == o.v_; }
    constexpr bool operator!=(Trit o) const noexcept { return v_ != o.v_; }

private:
    int8_t v_;
};

inline constexpr Trit T_NEG{-1};
inline constexpr Trit T_ZERO{0};
inline constexpr Trit T_POS{+1};

struct TritAdd { Trit sum; Trit carry; };

constexpr TritAdd trit_add(Trit a, Trit b) noexcept {
    int s = a.value() + b.value();
    int carry = 0, sum = s;
    if (s >= 2)       { sum -= 3; carry = +1; }
    else if (s <= -2) { sum += 3; carry = -1; }
    return {Trit{sum}, Trit{carry}};
}

constexpr TritAdd trit_full_add(Trit a, Trit b, Trit c) noexcept {
    int s = a.value() + b.value() + c.value();
    int carry = 0, sum = s;
    if (s >= 2)       { sum -= 3; carry = +1; }
    else if (s <= -2) { sum += 3; carry = -1; }
    return {Trit{sum}, Trit{carry}};
}

constexpr Trit trit_max(Trit a, Trit b) noexcept {
    return Trit{a.value() > b.value() ? a.value() : b.value()};
}
constexpr Trit trit_min(Trit a, Trit b) noexcept {
    return Trit{a.value() < b.value() ? a.value() : b.value()};
}

}
