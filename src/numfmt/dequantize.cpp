#include <ter/numfmt.hpp>

namespace ter {

void dequantize(const TritTensor& t, float* out) {
    size_t n = t.num_elems();
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(t.payload[i]) * t.scale;
    }
}

}
