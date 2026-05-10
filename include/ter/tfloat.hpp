#pragma once
#include <cstdint>
#include <cmath>

namespace ter {

// Format A (tfloat): 1 trit sign + 5 trits exponent + 9 trits mantissa = 15 trits.
// value = sign * |mantissa| * 3^exp,  where mantissa is balanced-ternary in
// [-9841, 9841] and exp is in [-121, 121]. Sign is encoded redundantly with
// the mantissa's own sign for symmetry; in this MVP we fold sign into mantissa
// and reserve the sign trit for future use (e.g., NaN flags, signed zero).
//
// Encode strategy: keep mantissa as close to the upper-magnitude band
// (|m| in [3281, 9841]) for max precision, by adjusting exp.
struct TFloat {
    int8_t  sign     = 0;     // -1, 0, +1   (0 means "use mantissa sign / zero")
    int8_t  exp      = 0;     // [-121, 121]
    int16_t mantissa = 0;     // [-9841, 9841]

    static constexpr int16_t MANT_MAX = 9841;   // 3^9 / 2 (default 9-trit mantissa)
    static constexpr int8_t  EXP_MAX  = 121;    // (3^5 - 1) / 2

    bool is_zero() const noexcept { return mantissa == 0; }

    // Variable-mantissa encoder: project x onto Format A with the given
    // mantissa-trit budget. mant_trits=9 is the default (15-trit total).
    // Smaller mantissa => fewer total trits per element but coarser precision.
    static TFloat from_float_trits(float x, int mant_trits) noexcept {
        TFloat t;
        if (!std::isfinite(x) || x == 0.0f || mant_trits <= 0) {
            return t;
        }
        // mant_max = (3^mant_trits - 1) / 2.
        long long m_max = 1;
        for (int i = 0; i < mant_trits; ++i) m_max *= 3;
        m_max = (m_max - 1) / 2;
        if (m_max > MANT_MAX) m_max = MANT_MAX;  // clamp to storage capacity

        double v = static_cast<double>(x);
        double absv = std::fabs(v);
        int exp = static_cast<int>(std::floor(std::log(absv) / std::log(3.0))) - (mant_trits - 1);
        double m = v / std::pow(3.0, static_cast<double>(exp));
        while (std::fabs(m) > static_cast<double>(m_max) && exp < EXP_MAX) {
            ++exp;
            m = v / std::pow(3.0, static_cast<double>(exp));
        }
        while (std::fabs(m) < static_cast<double>(m_max) / 3.0 && exp > -EXP_MAX) {
            --exp;
            m = v / std::pow(3.0, static_cast<double>(exp));
        }
        if (exp >  EXP_MAX) exp =  EXP_MAX;
        if (exp < -EXP_MAX) exp = -EXP_MAX;
        long mr = std::lround(v / std::pow(3.0, static_cast<double>(exp)));
        if (mr >  m_max) mr =  m_max;
        if (mr < -m_max) mr = -m_max;
        t.exp = static_cast<int8_t>(exp);
        t.mantissa = static_cast<int16_t>(mr);
        return t;
    }

    // Encode an arbitrary float into Format A. Round-to-nearest. Out-of-range
    // values clamp to ±max-representable. Uses the default 9-trit mantissa.
    static TFloat from_float(float x) noexcept {
        TFloat t;
        if (!std::isfinite(x) || x == 0.0f) {
            t.sign = 0;
            t.exp  = 0;
            t.mantissa = 0;
            return t;
        }
        double v = static_cast<double>(x);
        // Choose exp so that |v / 3^exp| is in [MANT_MAX/3, MANT_MAX].
        double absv = std::fabs(v);
        int exp = static_cast<int>(std::floor(std::log(absv) / std::log(3.0))) - 8;
        // Refine: nudge exp until mantissa fits the desired band.
        double m = v / std::pow(3.0, static_cast<double>(exp));
        while (std::fabs(m) > MANT_MAX && exp < EXP_MAX) {
            ++exp;
            m = v / std::pow(3.0, static_cast<double>(exp));
        }
        while (std::fabs(m) < MANT_MAX / 3.0 && exp > -EXP_MAX) {
            --exp;
            m = v / std::pow(3.0, static_cast<double>(exp));
        }
        // Clamp exp to [-EXP_MAX, EXP_MAX]; recompute mantissa.
        if (exp >  EXP_MAX) exp =  EXP_MAX;
        if (exp < -EXP_MAX) exp = -EXP_MAX;
        long mr = std::lround(v / std::pow(3.0, static_cast<double>(exp)));
        if (mr >  MANT_MAX) mr =  MANT_MAX;
        if (mr < -MANT_MAX) mr = -MANT_MAX;
        t.sign = 0;
        t.exp  = static_cast<int8_t>(exp);
        t.mantissa = static_cast<int16_t>(mr);
        return t;
    }

    float to_float() const noexcept {
        if (mantissa == 0) return 0.0f;
        double v = static_cast<double>(mantissa) * std::pow(3.0, static_cast<double>(exp));
        return static_cast<float>(v);
    }
};

}  // namespace ter
