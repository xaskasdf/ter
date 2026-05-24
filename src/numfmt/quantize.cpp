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
        t.payload[i] = static_cast<int32_t>(r);
    }
    return t;
}

TritTensor quantize_blocked(const float* data, const std::vector<int>& shape,
                            int n_trits_per_elem, int block_size) {
    if (n_trits_per_elem < 1 || n_trits_per_elem > Word27::kTrits) {
        throw std::out_of_range("quantize_blocked: n_trits_per_elem must be in [1, 27]");
    }
    if (block_size < 1) {
        throw std::out_of_range("quantize_blocked: block_size must be >= 1");
    }
    TritTensor t;
    t.dtype = DType::TritFP_B;
    t.n_trits_per_elem = n_trits_per_elem;
    t.block_size = block_size;
    t.shape = shape;

    size_t n = t.num_elems();
    t.payload.resize(n);
    const int64_t mti = max_trit_int(n_trits_per_elem);

    size_t n_blocks = (n + static_cast<size_t>(block_size) - 1)
                      / static_cast<size_t>(block_size);
    t.block_scales.resize(n_blocks, 0.0f);

    for (size_t b = 0; b < n_blocks; ++b) {
        size_t lo = b * static_cast<size_t>(block_size);
        size_t hi = lo + static_cast<size_t>(block_size);
        if (hi > n) hi = n;

        float max_abs = 0.0f;
        for (size_t i = lo; i < hi; ++i) {
            float a = std::fabs(data[i]);
            if (a > max_abs) max_abs = a;
        }
        if (max_abs == 0.0f) {
            t.block_scales[b] = 0.0f;
            for (size_t i = lo; i < hi; ++i) t.payload[i] = 0;
            continue;
        }
        const float sc = max_abs / static_cast<float>(mti);
        t.block_scales[b] = sc;
        for (size_t i = lo; i < hi; ++i) {
            int64_t r = static_cast<int64_t>(std::lround(data[i] / sc));
            if (r > mti)  r = mti;
            if (r < -mti) r = -mti;
            t.payload[i] = static_cast<int32_t>(r);
        }
    }
    return t;
}

}
