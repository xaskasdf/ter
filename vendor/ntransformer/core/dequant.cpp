#include <core/dequant.hpp>
#include <cstring>

namespace nt {

namespace {

// IEEE 754 half-to-float, branchless, no NaN/Inf special-casing beyond the obvious.
// Adapted from the standard "manual" implementation; safe for all bit patterns.
float half_to_float(std::uint16_t h) {
    std::uint32_t sign = (h & 0x8000u) << 16;
    std::uint32_t exp  = (h >> 10) & 0x1fu;
    std::uint32_t mant = h & 0x3ffu;
    std::uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;                             // ±0
        } else {
            // Subnormal: renormalize into a normal f32
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; --exp; }
            mant &= 0x3ffu;
            f = sign | ((127u - 15u - 1u + exp) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        f = sign | 0x7f800000u | (mant << 13);    // ±Inf or NaN
    } else {
        f = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

}  // namespace

void dequant_f16(const void* src, std::size_t n_elems, float* out) {
    const std::uint16_t* halves = static_cast<const std::uint16_t*>(src);
    for (std::size_t i = 0; i < n_elems; ++i) out[i] = half_to_float(halves[i]);
}

void dequant_f32(const void* src, std::size_t n_elems, float* out) {
    std::memcpy(out, src, n_elems * sizeof(float));
}

}  // namespace nt
