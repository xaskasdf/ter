#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/tx/layer.hpp>
#include <vector>
#include <random>

using namespace ter;
using namespace ter::tx;

TEST_CASE("quantize_layer fills all 7 weight tensors") {
    constexpr int H = 4, HD = 4, I = 8;
    std::vector<float> Wq(H * HD), Wk(H * HD), Wv(H * HD), Wo(HD * H);
    std::vector<float> Wgate(H * I), Wup(H * I), Wdown(I * H);
    std::vector<float> nw1(H, 1.0f), nw2(H, 1.0f);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : Wq)    v = dist(rng);
    for (auto& v : Wk)    v = dist(rng);
    for (auto& v : Wv)    v = dist(rng);
    for (auto& v : Wo)    v = dist(rng);
    for (auto& v : Wgate) v = dist(rng);
    for (auto& v : Wup)   v = dist(rng);
    for (auto& v : Wdown) v = dist(rng);

    LayerWeights L = quantize_layer(
        Wq.data(),    H,  HD,
        Wk.data(),    H,  HD,
        Wv.data(),    H,  HD,
        Wo.data(),    HD, H,
        Wgate.data(), H,  I,
        Wup.data(),   H,  I,
        Wdown.data(), I,  H,
        nw1.data(),   H,
        nw2.data(),   H);

    CHECK(L.Wq.payload.size()    == static_cast<size_t>(H * HD));
    CHECK(L.Wk.payload.size()    == static_cast<size_t>(H * HD));
    CHECK(L.Wv.payload.size()    == static_cast<size_t>(H * HD));
    CHECK(L.Wo.payload.size()    == static_cast<size_t>(HD * H));
    CHECK(L.Wgate.payload.size() == static_cast<size_t>(H * I));
    CHECK(L.Wup.payload.size()   == static_cast<size_t>(H * I));
    CHECK(L.Wdown.payload.size() == static_cast<size_t>(I * H));
    CHECK(L.Wq.scale > 0.0f);
    CHECK(L.attn_norm_w.size() == static_cast<size_t>(H));
    CHECK(L.ffn_norm_w.size()  == static_cast<size_t>(H));
}
