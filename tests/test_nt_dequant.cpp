#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <core/dequant.hpp>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cstring>

using namespace nt;

TEST_CASE("dequant_f16 handles known bit patterns") {
    std::vector<std::uint16_t> halves = {
        0x0000,   // +0
        0x8000,   // -0
        0x3c00,   // +1.0
        0xbc00,   // -1.0
        0x4200,   // +3.0
        0x3555,   // ~0.333
    };
    std::vector<float> out(halves.size());
    dequant_f16(halves.data(), halves.size(), out.data());
    CHECK(out[0] == 0.0f);
    CHECK(out[1] == -0.0f);
    CHECK(out[2] == 1.0f);
    CHECK(out[3] == -1.0f);
    CHECK(out[4] == 3.0f);
    CHECK(std::fabs(out[5] - 0.33325195f) < 1e-5f);
}

TEST_CASE("dequant_f32 is identity") {
    std::vector<float> in = {1.0f, -2.5f, 3.14f};
    std::vector<float> out(in.size());
    dequant_f32(in.data(), in.size(), out.data());
    for (std::size_t i = 0; i < in.size(); ++i) CHECK(out[i] == in[i]);
}

TEST_CASE("dequant_i2_s channel-interleaved 128-block with per-tensor scale") {
    // microsoft/BitNet i2_s layout (validated: BitNet 2B-4T forward is 8/8
    // bit-exact vs llama-cli with this decode):
    //   - flat row-major elements in BLOCKS of 128 -> 32 code bytes per block.
    //   - byte p (p=0..31) holds 4 codes for block positions {p,p+32,p+64,p+96};
    //     group g (position p + g*32) at bit shift (6 - g*2), group 0 in HIGH bits.
    //   - code map: 0 -> -1, 1 -> 0, 2 -> +1, 3 -> 0.
    //   - per-tensor scale appended as fp32 right after the codes (byte n_elems/4).
    constexpr int N = 128;
    std::uint8_t buf[32 + sizeof(float)];
    // All code bytes = 0b00_01_10_11 = 0x1B:
    //   g0 (shift 6) = 0 -> -1   (positions  0..31)
    //   g1 (shift 4) = 1 ->  0   (positions 32..63)
    //   g2 (shift 2) = 2 -> +1   (positions 64..95)
    //   g3 (shift 0) = 3 ->  0   (positions 96..127)
    for (int p = 0; p < 32; ++p) buf[p] = 0x1B;
    const float scale = 2.0f;
    std::memcpy(buf + 32, &scale, sizeof(float));
    float out[N];
    dequant_i2_s(buf, N, out);
    for (int i = 0;  i < 32;  ++i) CHECK(out[i] == -2.0f);
    for (int i = 32; i < 64;  ++i) CHECK(out[i] ==  0.0f);
    for (int i = 64; i < 96;  ++i) CHECK(out[i] == +2.0f);
    for (int i = 96; i < 128; ++i) CHECK(out[i] ==  0.0f);
}

TEST_CASE("dequant_q6_k on a hand-crafted constant block") {
    // d=1.0, all scales=1, all q nibbles=0 -> elem = 1 * 1 * (0|0 - 32) = -32.
    std::vector<std::uint8_t> blk(210, 0);
    for (int i = 0; i < 16; ++i) blk[192 + i] = 1;  // scales
    std::uint16_t d_h = 0x3C00;
    std::memcpy(&blk[208], &d_h, 2);

    std::vector<float> out(256);
    dequant_q6_k(blk.data(), 256, out.data());
    for (std::size_t i = 0; i < 256; ++i) {
        CHECK(out[i] == -32.0f);
    }
}

TEST_CASE("dequant_q4_k_m on a hand-crafted constant block") {
    // Build one block_q4_K of 256 elements, all == d * scale * 15 - dmin * 0 = 150.
    // d = dmin = 1.0 (fp16 0x3C00); each sub-block scale = 10, min = 0.
    // qs nibbles all = 15 (0xFF byte).
    std::vector<std::uint8_t> blk(144, 0);
    std::uint16_t d_h = 0x3C00, dmin_h = 0x3C00;
    std::memcpy(&blk[0], &d_h, 2);
    std::memcpy(&blk[2], &dmin_h, 2);
    // scales[0..3] = scale 10, scales[4..7] = min 0, scales[8..11] = packed (sub-block 4..7
    // takes low nibble = 10, high 2 bits from q[0..3] >> 6 = 0 since 10 < 64).
    for (int i = 0; i < 4; ++i) blk[4 + i]     = 10;
    for (int i = 0; i < 4; ++i) blk[4 + 4 + i] = 0;
    for (int i = 0; i < 4; ++i) blk[4 + 8 + i] = 0xA;  // low nibble 10, high nibble (min) = 0
    // qs[0..127] = 0xFF (every nibble = 15).
    for (int i = 0; i < 128; ++i) blk[16 + i] = 0xFF;

    std::vector<float> out(256);
    dequant_q4_k_m(blk.data(), 256, out.data());
    for (std::size_t i = 0; i < 256; ++i) {
        CHECK(out[i] == 150.0f);
    }
}
