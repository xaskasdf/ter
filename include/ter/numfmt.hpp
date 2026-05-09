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
    float scale = 0.0f;
    std::vector<int> shape;
    std::vector<int32_t> payload;

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

void dequantize(const TritTensor& t, float* out);

}
