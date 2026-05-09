#include <ter/vec.hpp>
#include <algorithm>
#include <limits>

namespace ter {

Vec vec_add(const Vec& a, const Vec& b) noexcept {
    Vec r;
    for (int i = 0; i < Vec::kLanes; ++i) {
        int64_t s = int64_t{a.lane(i)} + int64_t{b.lane(i)};
        r.set_lane_wide(i, static_cast<int32_t>(std::clamp<int64_t>(s,
            std::numeric_limits<int32_t>::min(),
            std::numeric_limits<int32_t>::max())));
    }
    return r;
}

Vec vec_sub(const Vec& a, const Vec& b) noexcept {
    Vec r;
    for (int i = 0; i < Vec::kLanes; ++i) {
        int64_t s = int64_t{a.lane(i)} - int64_t{b.lane(i)};
        r.set_lane_wide(i, static_cast<int32_t>(std::clamp<int64_t>(s,
            std::numeric_limits<int32_t>::min(),
            std::numeric_limits<int32_t>::max())));
    }
    return r;
}

Vec vec_neg(const Vec& a) noexcept {
    Vec r;
    for (int i = 0; i < Vec::kLanes; ++i) {
        // -INT32_MIN overflows; clamp to INT32_MAX.
        int32_t v = a.lane(i);
        if (v == std::numeric_limits<int32_t>::min()) v = std::numeric_limits<int32_t>::max();
        else v = -v;
        r.set_lane_wide(i, v);
    }
    return r;
}

Vec vec_mul(const Vec& a, const Vec& b) noexcept {
    // Lane-wise integer multiply.  Uses set_lane_wide so that scaled
    // fixed-point products (e.g. from the rsqrt LUT in tk_rmsnorm) are stored
    // without premature saturation at kLaneMax.  The caller is responsible for
    // applying a recovery scale to bring values back into the normal range.
    Vec r;
    for (int i = 0; i < Vec::kLanes; ++i) {
        int64_t prod = int64_t{a.lane(i)} * int64_t{b.lane(i)};
        r.set_lane_wide(i, prod);
    }
    return r;
}

Vec vec_broadcast(int32_t v) noexcept {
    Vec r;
    for (int i = 0; i < Vec::kLanes; ++i) r.set_lane(i, v);
    return r;
}

void vec_mac(VAccum& acc, const Vec& a, const Vec& b) noexcept {
    for (int i = 0; i < Vec::kLanes; ++i)
        acc.lanes[i] += int64_t{a.lane(i)} * int64_t{b.lane(i)};
}

int64_t vec_sum(const VAccum& acc) noexcept {
    int64_t s = 0;
    for (int i = 0; i < Vec::kLanes; ++i) s += acc.lanes[i];
    return s;
}

}
