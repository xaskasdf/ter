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

void dequant_q8_0(const void* src, std::size_t n_elems, float* out) {
    constexpr std::size_t QK = 32;
    const std::uint8_t* bytes = static_cast<const std::uint8_t*>(src);
    std::size_t n_blocks = n_elems / QK;
    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::uint8_t* blk = bytes + b * 34;
        std::uint16_t d_h;
        std::memcpy(&d_h, blk, sizeof(d_h));
        float d = half_to_float(d_h);
        const std::int8_t* qs = reinterpret_cast<const std::int8_t*>(blk + 2);
        for (std::size_t i = 0; i < QK; ++i) {
            out[b * QK + i] = d * static_cast<float>(qs[i]);
        }
    }
}

namespace {
// Per ggml block_q4_K convention: extract the 6-bit scale and min for sub-block j
// (j in 0..7) from the packed 12-byte `scales` array. Mirrors get_scale_min_k4
// from ggml-quants.c.
inline void get_scale_min_k4(int j, const std::uint8_t* q,
                             std::uint8_t* d, std::uint8_t* m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}
}  // namespace

void dequant_q6_k(const void* src, std::size_t n_elems, float* out) {
    constexpr std::size_t QK_K = 256;
    constexpr std::size_t BLK_BYTES = 210;
    const std::uint8_t* bytes = static_cast<const std::uint8_t*>(src);
    std::size_t n_blocks = n_elems / QK_K;
    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::uint8_t* blk = bytes + b * BLK_BYTES;
        const std::uint8_t* ql = blk;                              // 128 bytes
        const std::uint8_t* qh = blk + 128;                        // 64 bytes
        const std::int8_t*  sc = reinterpret_cast<const std::int8_t*>(blk + 192);  // 16 bytes
        std::uint16_t d_h;
        std::memcpy(&d_h, blk + 208, sizeof(d_h));
        float d = half_to_float(d_h);

        float* y = out + b * QK_K;
        // Process two 128-elem halves; ggml layout interleaves ql/qh in groups of 32.
        for (std::size_t base = 0; base < QK_K; base += 128) {
            const std::uint8_t* ql_h = ql + (base / 128) * 64;
            const std::uint8_t* qh_h = qh + (base / 128) * 32;
            const std::int8_t*  sc_h = sc + (base / 128) * 8;
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                std::int8_t q1 = static_cast<std::int8_t>((ql_h[l +  0] & 0xF) | (((qh_h[l] >> 0) & 3) << 4)) - 32;
                std::int8_t q2 = static_cast<std::int8_t>((ql_h[l + 32] & 0xF) | (((qh_h[l] >> 2) & 3) << 4)) - 32;
                std::int8_t q3 = static_cast<std::int8_t>((ql_h[l +  0] >>  4) | (((qh_h[l] >> 4) & 3) << 4)) - 32;
                std::int8_t q4 = static_cast<std::int8_t>((ql_h[l + 32] >>  4) | (((qh_h[l] >> 6) & 3) << 4)) - 32;
                y[base + l +  0] = d * static_cast<float>(sc_h[is + 0]) * static_cast<float>(q1);
                y[base + l + 32] = d * static_cast<float>(sc_h[is + 2]) * static_cast<float>(q2);
                y[base + l + 64] = d * static_cast<float>(sc_h[is + 4]) * static_cast<float>(q3);
                y[base + l + 96] = d * static_cast<float>(sc_h[is + 6]) * static_cast<float>(q4);
            }
        }
    }
}

void dequant_i2_s(const void* src, std::size_t n_elems, float* out) {
    // microsoft/BitNet i2_s layout (matches tools/convert_bitnet_to_packed.py,
    // validated against the 8/8-correct CUDA forward):
    //   - flat row-major elements in BLOCKS of 128 → 32 bytes per block.
    //   - byte p (p=0..31) holds 4 codes for block positions {p, p+32, p+64,
    //     p+96}; group g (position p + g*32) lives at bit shift (6 - g*2)
    //     (i.e. group 0 in the HIGH bits).
    //   - code map: 0 -> -1, 1 -> 0, 2 -> +1, 3 -> 0.
    // Per-tensor scale: microsoft appends 8 fp32 (32 bytes) right after the
    // codes; the first is the i2_scale (codes occupy n_elems/4 bytes). The
    // model is mmap'd full-file so the trailer is in-bounds. Applying it here
    // recovers the real weight magnitudes (scale × {-1,0,+1}).
    static const float LUT[4] = { -1.0f, 0.0f, +1.0f, 0.0f };
    const std::uint8_t* bytes = static_cast<const std::uint8_t*>(src);
    float i2_scale = 1.0f;
    std::memcpy(&i2_scale, bytes + n_elems / 4, sizeof(float));
    const std::size_t n_blocks = n_elems / 128;
    for (std::size_t blk = 0; blk < n_blocks; ++blk) {
        const std::uint8_t* bb = bytes + blk * 32;
        for (int p = 0; p < 32; ++p) {
            const std::uint8_t v = bb[p];
            for (int g = 0; g < 4; ++g) {
                const int shift = 6 - g * 2;
                out[blk * 128 + static_cast<std::size_t>(g) * 32
                    + static_cast<std::size_t>(p)] = LUT[(v >> shift) & 3] * i2_scale;
            }
        }
    }
    // Tail (n_elems not a multiple of 128): zero-fill. BitNet weight dims are
    // all multiples of 128, so this is defensive only.
    for (std::size_t i = n_blocks * 128; i < n_elems; ++i) out[i] = 0.0f;
}

void dequant_q4_k_m(const void* src, std::size_t n_elems, float* out) {
    constexpr std::size_t QK_K = 256;
    constexpr std::size_t BLK_BYTES = 144;
    const std::uint8_t* bytes = static_cast<const std::uint8_t*>(src);
    std::size_t n_blocks = n_elems / QK_K;
    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::uint8_t* blk = bytes + b * BLK_BYTES;
        std::uint16_t d_h, dmin_h;
        std::memcpy(&d_h,    blk,     sizeof(d_h));
        std::memcpy(&dmin_h, blk + 2, sizeof(dmin_h));
        float d    = half_to_float(d_h);
        float dmin = half_to_float(dmin_h);

        const std::uint8_t* scales = blk + 4;       // 12 bytes
        const std::uint8_t* qs     = blk + 4 + 12;  // 128 bytes

        float* y = out + b * QK_K;
        int is = 0;
        for (std::size_t base = 0; base < QK_K; base += 64) {
            std::uint8_t sc, mn;
            get_scale_min_k4(is + 0, scales, &sc, &mn);
            float d1 = d * static_cast<float>(sc);
            float m1 = dmin * static_cast<float>(mn);
            get_scale_min_k4(is + 1, scales, &sc, &mn);
            float d2 = d * static_cast<float>(sc);
            float m2 = dmin * static_cast<float>(mn);

            const std::uint8_t* q = qs + (base / 64) * 32;
            for (int l = 0; l < 32; ++l) y[base + l]      = d1 * static_cast<float>(q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) y[base + 32 + l] = d2 * static_cast<float>(q[l] >> 4)  - m2;
            is += 2;
        }
    }
}

}  // namespace nt
