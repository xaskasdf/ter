#pragma once
#include <ter/word.hpp>
#include <vector>
#include <cstdint>

namespace ter {

enum class DType : int {
    Float32  = 0,
    TritFP_B = 1,
};

// A tensor in format B: integer ternary payload + per-tensor float32 scale.
// Payload stores the integer ternary value directly (4 bytes/elem). Kernels that
// need a Word27 wrap on demand via Word27::from_int(payload[i]). Storing the
// full Word27 (~108 bytes) puts a 1B-param model past 100 GB; int32 keeps it ~4 GB.
struct TritTensor {
    DType dtype = DType::TritFP_B;
    int   n_trits_per_elem = 9;
    float scale = 0.0f;                  // per-tensor scale (used when block_size == 0)
    int   block_size = 0;                // 0 = per-tensor; >0 = per-block (Q8_0-style)
    std::vector<float> block_scales;     // size = ceil(num_elems / block_size) when block_size>0
    std::vector<int> shape;
    std::vector<int32_t> payload;
    // fp32 reference mode: when non-empty, holds the unquantized float weights
    // (payload is empty). Lets the forward run a true-FP path for validation
    // against an FP reference, isolating quantization from forward correctness.
    std::vector<float> fp32;

    size_t num_elems() const noexcept {
        if (shape.empty()) return 0;
        size_t n = 1;
        for (int d : shape) n *= static_cast<size_t>(d);
        return n;
    }
};

constexpr int64_t max_trit_int(int n_trits) noexcept {
    int64_t v = 1;
    for (int i = 0; i < n_trits; ++i) v *= 3;
    return (v - 1) / 2;
}

TritTensor quantize(const float* data,
                    const std::vector<int>& shape,
                    int n_trits_per_elem = 9);

// Per-block quantization (Q8_0-style): each contiguous group of `block_size`
// elements shares its own scale, instead of one scale for the whole tensor.
// Far less lossy when weights have outliers (a per-tensor scale dominated by
// one large weight crushes the rest). Populates block_scales + block_size.
TritTensor quantize_blocked(const float* data,
                            const std::vector<int>& shape,
                            int n_trits_per_elem,
                            int block_size);

void dequantize(const TritTensor& t, float* out);

}
