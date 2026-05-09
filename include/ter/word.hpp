#pragma once
#include <ter/trit.hpp>
#include <array>
#include <cstdint>

namespace ter {

class Word27 {
public:
    static constexpr int kTrits = 27;

    constexpr Word27() noexcept = default;

    static constexpr int64_t max_int() noexcept {
        int64_t v = 1;
        for (int i = 0; i < 27; ++i) v *= 3;
        return (v - 1) / 2;
    }
    static constexpr int64_t min_int() noexcept { return -max_int(); }

    static Word27 from_int(int64_t v) noexcept;
    int64_t to_int() const noexcept;

    constexpr Trit trit(int i) const noexcept { return trits_[i]; }
    constexpr void set_trit(int i, Trit t) noexcept { trits_[i] = t; }

    Word27 operator-() const noexcept;
    Word27 operator+(Word27 o) const noexcept;
    Word27 operator-(Word27 o) const noexcept;

    constexpr bool operator==(Word27 o) const noexcept {
        for (int i = 0; i < kTrits; ++i) if (trits_[i] != o.trits_[i]) return false;
        return true;
    }

private:
    std::array<Trit, kTrits> trits_{};
};

Trit sign_trit(Word27 w) noexcept;

}
