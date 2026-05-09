#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/numfmt.hpp>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <string>

#ifndef TER_KERNELS_DIR
#error "TER_KERNELS_DIR must be defined"
#endif

using namespace ter;

namespace {

constexpr int SEQ = 4;
constexpr int HID = 4;
constexpr int HEAD = 4;       // even for RoPE pairs
constexpr int VEC_LANES = 27;
constexpr int OUT_SCALE = 9841;

// Memory map (above kernel-code high-water mark, well above 512):
// A_BASE=1024, W_BASE=1300 reserved but addressed inline
constexpr int Q_BASE       = 1700;   // rope output scratch (reused per call)
constexpr int Y_TILE       = 2300;   // Y output of single-tile matmul
constexpr int SCRATCH_X    = 2400;   // gather buffer for matmul X
constexpr int SCRATCH_W    = 2500;   // gather buffer for matmul W column
constexpr int SCORES_BASE  = 2600;   // scores[s, t] padded (softmax x scratch)
constexpr int ATTN_BASE    = 2900;   // attn[s, t] post-softmax (softmax y scratch)
// CTX_BASE=3200, OUT_BASE=3500 reserved but addressed inline
constexpr int RCOS_BASE    = 3800;   // rope cos_vec
constexpr int RSIN_BASE    = 3900;
constexpr int RROT_BASE    = 4000;
constexpr int EXP_LUT_ADDR = 5000;
constexpr int RCP_LUT_ADDR = 6000;

// Helper: matmul one (1, K) row by a (K, N) matrix using tk_matmul_b_9t.
// X is given as a quantized TritTensor; W is given as a quantized TritTensor (K, N).
// Output is N float values.
void mm_row(Sim& s, KernelTable& kt, KernelId id_mm,
            const TritTensor& Xt, int row, const TritTensor& Wt,
            int K, int N,
            std::vector<float>& out) {
    out.assign(N, 0.0f);
    for (int j = 0; j < N; ++j) {
        int64_t int_acc = 0;
        for (int k0 = 0; k0 < K; k0 += 27) {
            int chunk = std::min(27, K - k0);
            for (int t = 0; t < 27; ++t) {
                int xv = (t < chunk) ? Xt.payload[row * K + (k0 + t)].to_int() : 0;
                int wv = (t < chunk) ? Wt.payload[(k0 + t) * N + j].to_int() : 0;
                s.mem().store_word(static_cast<size_t>(SCRATCH_X + t), Word27::from_int(xv));
                s.mem().store_word(static_cast<size_t>(SCRATCH_W + t), Word27::from_int(wv));
            }
            std::vector<int64_t> args = {SCRATCH_X, SCRATCH_W, Y_TILE, 0, 0, 0, 0};
            s.call_kernel(kt, id_mm, args);
            int_acc += s.mem().load_word(static_cast<size_t>(Y_TILE)).to_int();
        }
        out[j] = static_cast<float>(int_acc) * Xt.scale * Wt.scale;
    }
}

// Helper: apply tk_rope to a row of length HEAD (<= 26).
// in_xt is the quantized row (full VEC_LANES width). pos is the rotation position.
// Writes the rotated values back into out (float, length HEAD).
void rope_row(Sim& s, KernelTable& kt, KernelId id_rope,
              const TritTensor& xt_row, int pos,
              std::vector<float>& out) {
    constexpr int N_PAIRS = HEAD / 2;
    std::vector<int> cos_vec(VEC_LANES, 0);
    std::vector<int> sin_vec(VEC_LANES, 0);
    std::vector<int> rotated_x(VEC_LANES, 0);
    for (int k = 0; k < N_PAIRS; ++k) {
        double freq = 1.0 / std::pow(10000.0, (2.0 * k) / double(HEAD));
        double angle = double(pos) * freq;
        int c_int = static_cast<int>(std::round(std::cos(angle) * OUT_SCALE));
        int s_int = static_cast<int>(std::round(std::sin(angle) * OUT_SCALE));
        cos_vec[2 * k]     = c_int;
        cos_vec[2 * k + 1] = c_int;
        sin_vec[2 * k]     = s_int;
        sin_vec[2 * k + 1] = s_int;
        int x0 = xt_row.payload[2 * k].to_int();
        int x1 = xt_row.payload[2 * k + 1].to_int();
        rotated_x[2 * k]     = -x1;
        rotated_x[2 * k + 1] = x0;
    }
    // Place x at scratch below Q_BASE
    int xtmp = Q_BASE - VEC_LANES;
    for (int i = 0; i < VEC_LANES; ++i) {
        s.mem().store_word(static_cast<size_t>(xtmp + i), xt_row.payload[i]);
        s.mem().store_word(static_cast<size_t>(RCOS_BASE + i), Word27::from_int(cos_vec[i]));
        s.mem().store_word(static_cast<size_t>(RSIN_BASE + i), Word27::from_int(sin_vec[i]));
        s.mem().store_word(static_cast<size_t>(RROT_BASE + i), Word27::from_int(rotated_x[i]));
    }
    std::vector<int64_t> args = {xtmp, RCOS_BASE, RSIN_BASE, RROT_BASE, Q_BASE, 0, 0};
    s.call_kernel(kt, id_rope, args);

    float recovery = xt_row.scale / static_cast<float>(OUT_SCALE);
    out.assign(HEAD, 0.0f);
    for (int i = 0; i < HEAD; ++i) {
        out[i] = static_cast<float>(s.mem().load_word(static_cast<size_t>(Q_BASE + i)).to_int()) * recovery;
    }
}

// Helper: tk_softmax of a row of length SEQ (padded into VEC_LANES with zeros).
// Returns float length SEQ. Reads exp_lut and rcp_lut already loaded into sim.
void softmax_row(Sim& s, KernelTable& kt, KernelId id_sm,
                 const std::vector<float>& scores_row,
                 std::vector<float>& out) {
    // Quantize scores into a 9-trit tensor.
    std::vector<float> padded(VEC_LANES, 0.0f);
    for (int i = 0; i < SEQ; ++i) padded[i] = scores_row[i];
    TritTensor st = quantize(padded.data(), {VEC_LANES}, 9);

    // Read x_step from meta (passed via global side-channel: read once, cache).
    static float x_step_cache = -1.0f;
    if (x_step_cache < 0.0f) {
        std::ifstream meta("lut_data/softmax_lut.meta");
        REQUIRE(meta.is_open());
        std::string line;
        while (std::getline(meta, line)) {
            if (line.rfind("x_step=", 0) == 0) x_step_cache = std::stof(line.substr(7));
        }
    }
    REQUIRE(x_step_cache > 0.0f);

    int x_addr = SCORES_BASE + 256;  // local scratch for the quantized scores
    int y_addr = ATTN_BASE + 256;
    for (int i = 0; i < VEC_LANES; ++i) s.mem().store_word(static_cast<size_t>(x_addr + i), st.payload[i]);

    int64_t x_scale_div = std::max<int64_t>(1, static_cast<int64_t>(std::round(x_step_cache / st.scale)));
    int64_t sum_div = (VEC_LANES * static_cast<int64_t>(OUT_SCALE)) / 255;

    std::vector<int64_t> args = {x_addr, y_addr, EXP_LUT_ADDR, RCP_LUT_ADDR,
                                 x_scale_div, sum_div, 0};
    s.call_kernel(kt, id_sm, args);

    // Read y_int. Recovery: renormalise on host (softmax output padded with zeros
    // inflates denominator, so renormalise to sum=1 over valid SEQ entries).
    out.assign(SEQ, 0.0f);
    double s_sum = 0.0;
    for (int i = 0; i < SEQ; ++i) {
        double yi = static_cast<double>(s.mem().load_word(static_cast<size_t>(y_addr + i)).to_int());
        out[i] = static_cast<float>(yi);
        s_sum += yi;
    }
    if (s_sum > 0.0) for (int i = 0; i < SEQ; ++i) out[i] = static_cast<float>(out[i] / s_sum);
}

}  // namespace

TEST_CASE("Single-head attention via kernel composition matches numpy") {
    // 1) Build random inputs and weights (float32).
    std::mt19937 rng(0xA77E);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> X(SEQ * HID);
    std::vector<float> Wq(HID * HEAD), Wk(HID * HEAD), Wv(HID * HEAD);
    std::vector<float> Wo(HEAD * HID);
    for (auto& v : X)  v = dist(rng);
    for (auto& v : Wq) v = dist(rng);
    for (auto& v : Wk) v = dist(rng);
    for (auto& v : Wv) v = dist(rng);
    for (auto& v : Wo) v = dist(rng);

    // 2) Numpy reference.
    auto numpy_attn = [&]() {
        // Q[s,h] = sum_d X[s,d]*Wq[d,h]
        auto matmul = [](const std::vector<float>& A, const std::vector<float>& B,
                         int M, int K, int N) {
            std::vector<float> C(M * N, 0.0f);
            for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j)
                for (int k = 0; k < K; ++k) C[i * N + j] += A[i * K + k] * B[k * N + j];
            return C;
        };
        auto Q = matmul(X, Wq, SEQ, HID, HEAD);
        auto K = matmul(X, Wk, SEQ, HID, HEAD);
        auto V = matmul(X, Wv, SEQ, HID, HEAD);

        // Apply RoPE per row.
        auto rope = [](std::vector<float>& M, int rows, int dim) {
            for (int s = 0; s < rows; ++s) {
                for (int k = 0; k < dim / 2; ++k) {
                    double freq = 1.0 / std::pow(10000.0, (2.0 * k) / double(dim));
                    double angle = double(s) * freq;
                    double c = std::cos(angle);
                    double si = std::sin(angle);
                    float x0 = M[s * dim + 2 * k], x1 = M[s * dim + 2 * k + 1];
                    M[s * dim + 2 * k]     = static_cast<float>(x0 * c - x1 * si);
                    M[s * dim + 2 * k + 1] = static_cast<float>(x0 * si + x1 * c);
                }
            }
        };
        rope(Q, SEQ, HEAD);
        rope(K, SEQ, HEAD);

        // scores[s,t] = (Q[s,:] . K[t,:]) / sqrt(HEAD)
        std::vector<float> scores(SEQ * SEQ, 0.0f);
        float scale = 1.0f / std::sqrt(float(HEAD));
        for (int s = 0; s < SEQ; ++s) for (int t = 0; t < SEQ; ++t) {
            double acc = 0.0;
            for (int h = 0; h < HEAD; ++h) acc += double(Q[s * HEAD + h]) * double(K[t * HEAD + h]);
            scores[s * SEQ + t] = static_cast<float>(acc) * scale;
        }
        // softmax per row
        std::vector<float> attn(SEQ * SEQ, 0.0f);
        for (int s = 0; s < SEQ; ++s) {
            double mx = *std::max_element(&scores[s * SEQ], &scores[s * SEQ + SEQ]);
            double sum_e = 0.0;
            std::vector<double> ex(SEQ);
            for (int t = 0; t < SEQ; ++t) { ex[t] = std::exp(double(scores[s * SEQ + t]) - mx); sum_e += ex[t]; }
            for (int t = 0; t < SEQ; ++t) attn[s * SEQ + t] = static_cast<float>(ex[t] / sum_e);
        }
        // ctx[s,h] = sum_t attn[s,t] * V[t,h]
        std::vector<float> ctx(SEQ * HEAD, 0.0f);
        for (int s = 0; s < SEQ; ++s) for (int h = 0; h < HEAD; ++h) {
            double acc = 0.0;
            for (int t = 0; t < SEQ; ++t) acc += double(attn[s * SEQ + t]) * double(V[t * HEAD + h]);
            ctx[s * HEAD + h] = static_cast<float>(acc);
        }
        // out = ctx @ Wo
        return matmul(ctx, Wo, SEQ, HEAD, HID);
    };
    std::vector<float> out_ref = numpy_attn();

    // 3) Sim setup and load LUTs.
    Sim s(8 * 1024);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    KernelId id_mm = kt.find("tk_matmul_b_9t");
    KernelId id_rope = kt.find("tk_rope");
    KernelId id_sm = kt.find("tk_softmax");
    REQUIRE(id_mm.valid);
    REQUIRE(id_rope.valid);
    REQUIRE(id_sm.valid);

    auto read_i32 = [](const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        REQUIRE(f.is_open());
        f.seekg(0, std::ios::end);
        size_t n = static_cast<size_t>(f.tellg()) / 4;
        f.seekg(0);
        std::vector<int32_t> v(n);
        f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4));
        return v;
    };

    auto exp_i32 = read_i32("lut_data/exp_lut.bin");
    auto rcp_i32 = read_i32("lut_data/rcp_lut.bin");
    std::vector<int> exp_lut(exp_i32.begin(), exp_i32.end());
    std::vector<int> rcp_lut(rcp_i32.begin(), rcp_i32.end());
    s.load_lut(EXP_LUT_ADDR, exp_lut);
    s.load_lut(RCP_LUT_ADDR, rcp_lut);

    // 4) Quantize X, Wq, Wk, Wv, Wo to format B.
    TritTensor Xt  = quantize(X.data(),  {SEQ, HID}, 9);
    TritTensor Wqt = quantize(Wq.data(), {HID, HEAD}, 9);
    TritTensor Wkt = quantize(Wk.data(), {HID, HEAD}, 9);
    TritTensor Wvt = quantize(Wv.data(), {HID, HEAD}, 9);
    TritTensor Wot = quantize(Wo.data(), {HEAD, HID}, 9);

    // 5) Compute Q, K, V via mm_row per row.
    std::vector<std::vector<float>> Q(SEQ), K(SEQ), V(SEQ);
    for (int sr = 0; sr < SEQ; ++sr) {
        mm_row(s, kt, id_mm, Xt, sr, Wqt, HID, HEAD, Q[sr]);
        mm_row(s, kt, id_mm, Xt, sr, Wkt, HID, HEAD, K[sr]);
        mm_row(s, kt, id_mm, Xt, sr, Wvt, HID, HEAD, V[sr]);
    }

    // 6) Apply RoPE to Q and K.
    for (int sr = 0; sr < SEQ; ++sr) {
        std::vector<float> qpad(VEC_LANES, 0.0f), kpad(VEC_LANES, 0.0f);
        for (int h = 0; h < HEAD; ++h) { qpad[h] = Q[sr][h]; kpad[h] = K[sr][h]; }
        TritTensor qt = quantize(qpad.data(), {VEC_LANES}, 9);
        TritTensor kt_row = quantize(kpad.data(), {VEC_LANES}, 9);
        rope_row(s, kt, id_rope, qt, sr, Q[sr]);
        rope_row(s, kt, id_rope, kt_row, sr, K[sr]);
    }

    // 7) scores[s,t] = (Q[s,:] . K[t,:]) / sqrt(HEAD) on host (HEAD tiny).
    float scale = 1.0f / std::sqrt(float(HEAD));
    std::vector<std::vector<float>> scores(SEQ, std::vector<float>(SEQ, 0.0f));
    for (int sr = 0; sr < SEQ; ++sr) for (int t = 0; t < SEQ; ++t) {
        double acc = 0.0;
        for (int h = 0; h < HEAD; ++h) acc += double(Q[sr][h]) * double(K[t][h]);
        scores[sr][t] = static_cast<float>(acc) * scale;
    }

    // 8) softmax per row via tk_softmax kernel.
    std::vector<std::vector<float>> attn(SEQ);
    for (int sr = 0; sr < SEQ; ++sr) softmax_row(s, kt, id_sm, scores[sr], attn[sr]);

    // 9) ctx[s,:] = sum_t attn[s,t] * V[t,:] (host op; tiny).
    std::vector<std::vector<float>> ctx(SEQ, std::vector<float>(HEAD, 0.0f));
    for (int sr = 0; sr < SEQ; ++sr) for (int h = 0; h < HEAD; ++h) {
        double acc = 0.0;
        for (int t = 0; t < SEQ; ++t) acc += double(attn[sr][t]) * double(V[t][h]);
        ctx[sr][h] = static_cast<float>(acc);
    }

    // 10) out[s,:] = ctx[s,:] @ Wo via mm_row.
    std::vector<std::vector<float>> out_(SEQ);
    for (int sr = 0; sr < SEQ; ++sr) {
        std::vector<float> ctx_pad(HEAD);
        for (int h = 0; h < HEAD; ++h) ctx_pad[h] = ctx[sr][h];
        // mm_row expects shape (M, K) for X and (K, N) for W; use M=1 by reshaping.
        TritTensor Xt_1 = quantize(ctx_pad.data(), {1, HEAD}, 9);
        mm_row(s, kt, id_mm, Xt_1, 0, Wot, HEAD, HID, out_[sr]);
    }

    // 11) Compare.
    double max_rel = 0.0;
    for (int sr = 0; sr < SEQ; ++sr) for (int h = 0; h < HID; ++h) {
        double ref = out_ref[sr * HID + h];
        double got = out_[sr][h];
        double denom = std::max(0.1, std::fabs(ref));
        double rel = std::fabs(got - ref) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    // Generous bound: chained quantization through 5 matmuls + softmax + rope compounds.
    // Acceptable threshold is "within 50% relative" — confirms correctness of orchestration,
    // not numerical fidelity. A real-precision attention pass would need higher trit counts.
    CHECK(max_rel < 0.5);

    // Counter sanity: many TVMACs (one per matmul tile, many tiles), TVMUL from rope,
    // TLOADs from softmax. Just confirm none are zero.
    CHECK(s.counters().get(Opcode::TVMAC)  > 0);
    CHECK(s.counters().get(Opcode::TVMUL)  > 0);
    CHECK(s.counters().get(Opcode::TLOAD)  > 0);
    CHECK(s.counters().get(Opcode::TVLOAD) > 0);
}
