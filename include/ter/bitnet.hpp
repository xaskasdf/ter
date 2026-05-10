#pragma once
#include <cstdint>
#include <cstddef>
#include <cmath>

namespace ter {

// BitNet b1.58: weights live in {-1, 0, +1} only. The quantizer follows the
// paper's recipe: scale = mean(|x|); q = round(x / scale) clamped to [-1, +1].
// Returns the per-tensor scale used; out[] is filled with int8 ternary digits.
inline float quantize_bitnet(const float* in, std::size_t n, std::int8_t* out) {
    if (n == 0) return 0.0f;
    double sum_abs = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum_abs += std::fabs(in[i]);
    double scale = sum_abs / static_cast<double>(n);
    if (scale == 0.0) {
        for (std::size_t i = 0; i < n; ++i) out[i] = 0;
        return 0.0f;
    }
    for (std::size_t i = 0; i < n; ++i) {
        double q = static_cast<double>(in[i]) / scale;
        long r = std::lround(q);
        if (r >  1) r =  1;
        if (r < -1) r = -1;
        out[i] = static_cast<std::int8_t>(r);
    }
    return static_cast<float>(scale);
}

// Analytical op-count for a BitNet matmul: with weights in {-1, 0, +1}, every
// multiply collapses to a sign-flip-or-skip plus an add. No TVMAC; the inner
// loop is pure TVADD (and TVSUB for negatives). Compared to Format B (which
// emits ceil(K/27) TVMACs per output cell), BitNet emits at most K TVADDs per
// output cell minus the zeros. For an "honest" comparison vs analytical fp16
// MACs, BitNet substitutes ADDs for MACs entirely.
struct BitNetMatmulOps {
    std::uint64_t tvadd = 0;   // non-zero positive weights
    std::uint64_t tvsub = 0;   // non-zero negative weights
    std::uint64_t skips = 0;   // zero weights (no op emitted)
    std::uint64_t tvmac = 0;   // always 0 by design
};

inline BitNetMatmulOps bitnet_matmul_ops(const std::int8_t* W, std::size_t K, std::size_t N) {
    BitNetMatmulOps c;
    for (std::size_t i = 0; i < K * N; ++i) {
        if (W[i] >  0) ++c.tvadd;
        else if (W[i] < 0) ++c.tvsub;
        else ++c.skips;
    }
    return c;
}

}  // namespace ter
