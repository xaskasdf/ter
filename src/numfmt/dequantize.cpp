#include <ter/numfmt.hpp>

namespace ter {

void dequantize(const TritTensor& t, float* out) {
    size_t n = t.num_elems();
    // fp32 reference mode: weights stored directly as float.
    if (!t.fp32.empty()) {
        for (size_t i = 0; i < n; ++i) out[i] = t.fp32[i];
        return;
    }
    // per-block scales (Q8_0-style).
    if (!t.block_scales.empty()) {
        const size_t bs = static_cast<size_t>(t.block_size);
        for (size_t i = 0; i < n; ++i)
            out[i] = static_cast<float>(t.payload[i]) * t.block_scales[i / bs];
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(t.payload[i]) * t.scale;
    }
}

}
