#pragma once
#include <array>
#include <cstdint>

namespace ter {

class Vec {
public:
    static constexpr int kLanes = 27;
    static constexpr int kTritsPerLane = 9;
    static constexpr int kLaneMin = -9841;
    static constexpr int kLaneMax = +9841;

    constexpr int lane(int i) const noexcept { return lanes_[i]; }
    constexpr void set_lane(int i, int v) noexcept {
        lanes_[i] = v < kLaneMin ? kLaneMin : v > kLaneMax ? kLaneMax : v;
    }

private:
    std::array<int32_t, kLanes> lanes_{};
};

Vec vec_add(const Vec& a, const Vec& b) noexcept;
Vec vec_sub(const Vec& a, const Vec& b) noexcept;
Vec vec_neg(const Vec& a) noexcept;
Vec vec_broadcast(int32_t v) noexcept;

struct VAccum { std::array<int64_t, Vec::kLanes> lanes{}; };

void vec_mac(VAccum& acc, const Vec& a, const Vec& b) noexcept;
int64_t vec_sum(const VAccum& acc) noexcept;

}
