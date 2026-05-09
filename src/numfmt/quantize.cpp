#include <ter/numfmt.hpp>
#include <cmath>
#include <stdexcept>

namespace ter {

TritTensor quantize(const float* data, const std::vector<int>& shape, int n_trits_per_elem) {
    if (n_trits_per_elem < 1 || n_trits_per_elem > Word27::kTrits) {
        throw std::out_of_range("quantize: n_trits_per_elem must be in [1, 27]");
    }
    TritTensor t;
    t.dtype = DType::TritFP_B;
    t.n_trits_per_elem = n_trits_per_elem;
    t.shape = shape;

    size_t n = t.num_elems();
    t.payload.resize(n);

    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float a = std::fabs(data[i]);
        if (a > max_abs) max_abs = a;
    }

    int64_t mti = max_trit_int(n_trits_per_elem);
    if (max_abs == 0.0f) {
        t.scale = 0.0f;
        return t;
    }

    t.scale = max_abs / static_cast<float>(mti);

    for (size_t i = 0; i < n; ++i) {
        float q = data[i] / t.scale;
        int64_t r = static_cast<int64_t>(std::lround(q));
        if (r > mti)  r = mti;
        if (r < -mti) r = -mti;
        t.payload[i] = Word27::from_int(r);
    }
    return t;
}

}
