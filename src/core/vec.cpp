#include <ter/vec.hpp>

namespace ter {

Vec vec_add(const Vec& a, const Vec& b) noexcept {
    Vec r;
    for (int i = 0; i < Vec::kLanes; ++i) r.set_lane(i, a.lane(i) + b.lane(i));
    return r;
}

Vec vec_sub(const Vec& a, const Vec& b) noexcept {
    Vec r;
    for (int i = 0; i < Vec::kLanes; ++i) r.set_lane(i, a.lane(i) - b.lane(i));
    return r;
}

Vec vec_neg(const Vec& a) noexcept {
    Vec r;
    for (int i = 0; i < Vec::kLanes; ++i) r.set_lane(i, -a.lane(i));
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
