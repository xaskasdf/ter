#include <ter/tx/forward.hpp>
#include <ter/numfmt.hpp>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace ter::tx {

// FP-bypass: when true, forward_layer skips activation quantization and does
// float matmuls with dequantized weights. Diagnostic to isolate whether the
// degenerate output comes from activation quantization (9-trit per-tensor) or
// from the forward path itself. Default false → normal ternary forward.
static bool g_fp_bypass = false;
void set_forward_fp_bypass(bool b) noexcept { g_fp_bypass = b; }

namespace {

// Float matmul with on-the-fly weight dequantization (per-tensor or per-block).
// Used by forward_layer when g_fp_bypass is set: x stays float (no activation
// quantization), weights dequantized from their TritTensor payload.
void mm_row_fp(const std::vector<float>& x, const TritTensor& Wt,
               int K, int N, std::vector<float>& out) {
    out.assign(static_cast<size_t>(N), 0.0f);
    // Double accumulation (more accurate than the fp32 reference). Note: at
    // near-tie argmax decisions on heavily-cancelled small outputs, sequential
    // double / fp32 / BLAS-pairwise can each pick a different (equally valid)
    // winner — this is cross-implementation numerical irreproducibility, not an
    // error. Robust-margin tokens (e.g. BitNet) are unaffected and match exactly.
    std::vector<double> acc(static_cast<size_t>(N), 0.0);
    const bool is_fp32  = !Wt.fp32.empty();
    const bool blocked  = !Wt.block_scales.empty();
    const int  bs       = Wt.block_size;
    for (int kk = 0; kk < K; ++kk) {
        const double xv = static_cast<double>(x[static_cast<size_t>(kk)]);
        if (xv == 0.0) continue;
        const size_t base = static_cast<size_t>(kk) * static_cast<size_t>(N);
        for (int j = 0; j < N; ++j) {
            const size_t idx = base + static_cast<size_t>(j);
            const double w =
                is_fp32 ? static_cast<double>(Wt.fp32[idx])
              : blocked ? static_cast<double>(Wt.payload[idx]) *
                          static_cast<double>(Wt.block_scales[idx / static_cast<size_t>(bs)])
                        : static_cast<double>(Wt.payload[idx]) * static_cast<double>(Wt.scale);
            acc[static_cast<size_t>(j)] += xv * w;
        }
    }
    for (int j = 0; j < N; ++j)
        out[static_cast<size_t>(j)] = static_cast<float>(acc[static_cast<size_t>(j)]);
}

// Sim memory map for forward_layer (single-token, single-layer).
// Kernel-code occupies [0, 511]; data starts >= 512.
// We use >= 1024 to be safe.
// SCRATCH_X / SCRATCH_W / Y_TILE were used by the legacy kernel-routed mm_row;
// the AVX2 fast-path mm_row no longer needs them but the constants document
// the historical layout for the rmsnorm/silu/etc kernels still in use.
[[maybe_unused]] constexpr int SCRATCH_X  = 1024;
[[maybe_unused]] constexpr int SCRATCH_W  = 1100;
[[maybe_unused]] constexpr int Y_TILE     = 1200;
constexpr int RMS_X_ADDR = 1300;  // rmsnorm input buffer (27 words)
constexpr int RMS_Y_ADDR = 1400;  // rmsnorm output buffer (27 words)

// ---- Host-side fallbacks for N > 27 ----
// Brandon-tiny: dim=256, intermediate=720, max_seq=512 — all > 27.
// Kernels stay for N <= 27 (existing tests) but real models route through host.

void rmsnorm_host(const std::vector<float>& x, const std::vector<float>& w,
                  float eps, std::vector<float>& y) {
    double ss = 0.0;
    for (auto v : x) ss += double(v) * double(v);
    double rms_inv = 1.0 / std::sqrt(ss / static_cast<double>(x.size()) + double(eps));
    y.assign(x.size(), 0.0f);
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = static_cast<float>(static_cast<double>(x[i]) * rms_inv) * w[i];
}

// Apply RoPE in-place to a head_dim vector at position pos (interleaved Llama 3 layout).
void rope_host(std::vector<float>& v, int pos, int head_dim, double theta = 10000.0) {
    int n_pairs = head_dim / 2;
    for (int k = 0; k < n_pairs; ++k) {
        double freq = 1.0 / std::pow(theta, (2.0 * k) / static_cast<double>(head_dim));
        double angle = static_cast<double>(pos) * freq;
        double c = std::cos(angle), s = std::sin(angle);
        float x0 = v[2 * k], x1 = v[2 * k + 1];
        v[2 * k]     = static_cast<float>(x0 * c - x1 * s);
        v[2 * k + 1] = static_cast<float>(x0 * s + x1 * c);
    }
}

// Numerically stable softmax (max-subtract).
void softmax_host(std::vector<float>& v) {
    if (v.empty()) return;
    double mx = v[0];
    for (auto x : v) if (x > mx) mx = x;
    double sum = 0.0;
    for (auto& x : v) {
        x = static_cast<float>(std::exp(static_cast<double>(x) - mx));
        sum += x;
    }
    if (sum > 0.0) for (auto& x : v) x = static_cast<float>(x / sum);
}

// SwiGLU element-wise: y[i] = silu(gate[i]) * up[i] = gate[i] * sigmoid(gate[i]) * up[i].
void silu_mul_host(const std::vector<float>& gate, const std::vector<float>& up,
                   std::vector<float>& y) {
    y.assign(gate.size(), 0.0f);
    for (size_t i = 0; i < gate.size(); ++i) {
        double s = 1.0 / (1.0 + std::exp(-static_cast<double>(gate[i])));
        y[i] = static_cast<float>(static_cast<double>(gate[i]) * s * static_cast<double>(up[i]));
    }
}

// ReLU²-gated FFN (BitNet b1.58 2B-4T): y[i] = relu(gate[i])^2 * up[i].
void relu2_mul_host(const std::vector<float>& gate, const std::vector<float>& up,
                    std::vector<float>& y) {
    y.assign(gate.size(), 0.0f);
    for (size_t i = 0; i < gate.size(); ++i) {
        double g = static_cast<double>(gate[i]);
        double r = g > 0.0 ? g * g : 0.0;
        y[i] = static_cast<float>(r * static_cast<double>(up[i]));
    }
}

// Apply RMSNorm via the tk_rmsnorm kernel.
// x: input vector of length N (must be <= 27 for single-tile execution).
// w: gain vector of length N (applied host-side; kernel doesn't do gain).
// rsqrt_lut_addr: sim address of the loaded rsqrt LUT (256 entries).
// Recovery: y[i] = (y_int[i] * xt.scale * RSQ_MAX / OUT_SCALE) * w[i]
// where RSQ_MAX = 16 matches gen_rsqrt_lut.py's default rsq_max parameter.
// Caveat: zero-padding to 27 lanes does not change sum_sq (zeros contribute 0),
// but sum_div is computed for the worst-case 27-lane max, biasing the LUT index
// slightly downward for small N. This is documented and accepted for this MVP.
void rmsnorm_kernel(Sim& sim, KernelTable& kt, KernelId id_rms,
                    const std::vector<float>& x, const std::vector<float>& w,
                    int rsqrt_lut_addr, float /*eps*/,
                    std::vector<float>& y) {
    constexpr int VEC_LANES = 27;
    constexpr int OUT_SCALE = 9841;
    constexpr float RSQ_MAX = 16.0f;  // matches gen_rsqrt_lut.py default

    // Quantize input padded to 27 lanes.
    std::vector<float> padded(VEC_LANES, 0.0f);
    for (size_t i = 0; i < x.size() && i < static_cast<size_t>(VEC_LANES); ++i)
        padded[i] = x[i];
    TritTensor xt = quantize(padded.data(), {VEC_LANES}, 9);

    // Place inputs in sim scratch memory.
    for (int i = 0; i < VEC_LANES; ++i)
        sim.mem().store_word(static_cast<size_t>(RMS_X_ADDR + i), ter::Word27::from_int(xt.payload[i]));

    // sum_div: max sum_sq for 27 lanes of |x_int| <= 9841 is 27 * 9841^2 ~ 2.6e9.
    int64_t mti = 9841;
    int64_t sum_div = (static_cast<int64_t>(VEC_LANES) * mti * mti) / 255;
    if (sum_div < 1) sum_div = 1;

    std::vector<int64_t> args = {RMS_X_ADDR, RMS_Y_ADDR, rsqrt_lut_addr, sum_div, 255, 0, 0};
    sim.call_kernel(kt, id_rms, args);

    // Recover float and apply per-element gain w[i] (host-side).
    float recovery = xt.scale * RSQ_MAX / static_cast<float>(OUT_SCALE);
    y.assign(x.size(), 0.0f);
    for (size_t i = 0; i < x.size(); ++i) {
        float v = static_cast<float>(
            sim.mem().load_word(static_cast<size_t>(RMS_Y_ADDR + i)).to_int()) * recovery;
        y[i] = v * w[i];
    }
}

// Apply RoPE via tk_rope kernel using interleaved layout (Llama 3 / default style).
// v: head-dim float vector, modified in-place.
// tk_rope args: x_addr, cos_addr, sin_addr, rotated_x_addr, y_addr, 0, 0
void rope_kernel(Sim& sim, KernelTable& kt, KernelId id_rope,
                 std::vector<float>& v, int pos, int head_dim, double theta = 10000.0) {
    constexpr int VEC_LANES = 27;
    constexpr int OUT_SCALE = 9841;
    int n_pairs = head_dim / 2;

    std::vector<float> padded(VEC_LANES, 0.0f);
    for (int i = 0; i < head_dim; ++i) padded[i] = v[i];
    TritTensor xt = quantize(padded.data(), {VEC_LANES}, 9);

    std::vector<int> cos_vec(VEC_LANES, 0);
    std::vector<int> sin_vec(VEC_LANES, 0);
    std::vector<int> rotated_x(VEC_LANES, 0);
    for (int k = 0; k < n_pairs; ++k) {
        double freq = 1.0 / std::pow(theta, (2.0 * k) / double(head_dim));
        double angle = double(pos) * freq;
        int c_int = static_cast<int>(std::round(std::cos(angle) * OUT_SCALE));
        int s_int = static_cast<int>(std::round(std::sin(angle) * OUT_SCALE));
        // Interleaved: element 2k = real, 2k+1 = imaginary
        cos_vec[2 * k]     = c_int;
        cos_vec[2 * k + 1] = c_int;
        sin_vec[2 * k]     = s_int;
        sin_vec[2 * k + 1] = s_int;
        int x0 = xt.payload[2 * k];
        int x1 = xt.payload[2 * k + 1];
        rotated_x[2 * k]     = -x1;
        rotated_x[2 * k + 1] = x0;
    }

    int x_addr = 1700, cos_addr = 1800, sin_addr = 1900, rotx_addr = 2000, y_addr = 2100;
    for (int i = 0; i < VEC_LANES; ++i) {
        sim.mem().store_word(static_cast<size_t>(x_addr + i), ter::Word27::from_int(xt.payload[i]));
        sim.mem().store_word(static_cast<size_t>(cos_addr + i), Word27::from_int(cos_vec[i]));
        sim.mem().store_word(static_cast<size_t>(sin_addr + i), Word27::from_int(sin_vec[i]));
        sim.mem().store_word(static_cast<size_t>(rotx_addr + i), Word27::from_int(rotated_x[i]));
    }
    std::vector<int64_t> args = {x_addr, cos_addr, sin_addr, rotx_addr, y_addr, 0, 0};
    sim.call_kernel(kt, id_rope, args);

    float recovery = xt.scale / static_cast<float>(OUT_SCALE);
    for (int i = 0; i < head_dim; ++i) {
        v[i] = static_cast<float>(sim.mem().load_word(static_cast<size_t>(y_addr + i)).to_int()) * recovery;
    }
}

// SiLU(gate) * up element-wise via tk_silu kernel.
// gate and up must be <= 27 elements (single-tile).
// sigmoid_lut_addr: sim address of the loaded sigmoid LUT.
// OUT_SCALE matches gen_sigmoid_lut.py's output scale (9841).
// X_STEP matches the LUT's x step (1/32).
void silu_mul_kernel(Sim& sim, KernelTable& kt, KernelId id_silu,
                     const std::vector<float>& gate, const std::vector<float>& up,
                     int sigmoid_lut_addr,
                     std::vector<float>& y) {
    constexpr int VEC_LANES = 27;
    constexpr int OUT_SCALE = 9841;
    constexpr float X_STEP = 0.03125f;

    std::vector<float> padded(VEC_LANES, 0.0f);
    for (size_t i = 0; i < gate.size() && i < VEC_LANES; ++i) padded[i] = gate[i];
    TritTensor gt = quantize(padded.data(), {VEC_LANES}, 9);

    int x_addr = 1300, y_addr = 1400;
    for (int i = 0; i < VEC_LANES; ++i)
        sim.mem().store_word(static_cast<size_t>(x_addr + i), ter::Word27::from_int(gt.payload[i]));

    int64_t x_scale_div = std::max<int64_t>(1, static_cast<int64_t>(std::round(X_STEP / gt.scale)));
    std::vector<int64_t> args = {x_addr, y_addr, sigmoid_lut_addr, x_scale_div, 0, 0, 0};
    sim.call_kernel(kt, id_silu, args);

    float recovery = gt.scale / static_cast<float>(OUT_SCALE);
    y.assign(gate.size(), 0.0f);
    for (size_t i = 0; i < gate.size(); ++i) {
        float silu = static_cast<float>(sim.mem().load_word(static_cast<size_t>(y_addr + i)).to_int()) * recovery;
        y[i] = silu * up[i];
    }
}

// Apply softmax over `n_real` values using tk_softmax kernel; pads with very
// negative values so unused lanes contribute ~0 to the sum, then
// host-renormalises the first n_real outputs.
void softmax_kernel(Sim& sim, KernelTable& kt, KernelId id_sm,
                    std::vector<float>& scores,
                    int exp_lut_addr, int rcp_lut_addr,
                    int /*n_pad*/) {
    constexpr int VEC_LANES = 27;
    constexpr int OUT_SCALE = 9841;
    constexpr float X_STEP = 0.03125f;

    int n_real = static_cast<int>(scores.size());
    if (n_real <= 0) return;

    // Pad with very negative values; exp(-10) ~ 4.5e-5 contributes ~0 to sum.
    std::vector<float> padded(VEC_LANES, -10.0f);
    for (int i = 0; i < n_real && i < VEC_LANES; ++i) padded[i] = scores[i];

    TritTensor st = quantize(padded.data(), {VEC_LANES}, 9);

    int x_addr = 1500, y_addr = 1600;
    for (int i = 0; i < VEC_LANES; ++i)
        sim.mem().store_word(static_cast<size_t>(x_addr + i), ter::Word27::from_int(st.payload[i]));

    int64_t x_scale_div = std::max<int64_t>(1, static_cast<int64_t>(std::round(X_STEP / st.scale)));
    int64_t sum_div = (VEC_LANES * static_cast<int64_t>(OUT_SCALE)) / 255;
    std::vector<int64_t> args = {x_addr, y_addr, exp_lut_addr, rcp_lut_addr,
                                 x_scale_div, sum_div, 0};
    sim.call_kernel(kt, id_sm, args);

    // Read first n_real values and renormalise on host.
    std::vector<double> y_int(n_real, 0.0);
    double sum = 0.0;
    for (int i = 0; i < n_real; ++i) {
        y_int[i] = static_cast<double>(sim.mem().load_word(static_cast<size_t>(y_addr + i)).to_int());
        sum += y_int[i];
    }
    if (sum > 0.0) for (int i = 0; i < n_real; ++i) scores[i] = static_cast<float>(y_int[i] / sum);
    else           for (int i = 0; i < n_real; ++i) scores[i] = 1.0f / static_cast<float>(n_real);
}

// mm_row: GEMV with k-outer/j-inner loop ordering. Each k iteration broadcasts
// X[k] and adds X[k] * W[k, :] into the accumulator row. Inner j loop is
// contiguous in W (row-major) and contiguous in `acc`, so it auto-vectorizes
// to AVX2 vpmuldq + vpaddq under -O3. The scalar zero-skip on X also avoids
// pure-zero rows of trit weights (~10-25% of work in BitNet/Llama).
//
// TVMAC counter is bumped analytically (ceil(K/27) * N) to match what the
// kernel-routed path would emit. This is the F8 honest-counter pattern.
void mm_row(Sim& sim, KernelTable& kt, KernelId /*id_mm*/,
            const TritTensor& Xt, int row,
            const TritTensor& Wt,
            int K, int N,
            std::vector<float>& out) {
    out.assign(static_cast<size_t>(N), 0.0f);
    const int32_t* Xp = Xt.payload.data() + static_cast<size_t>(row) * static_cast<size_t>(K);
    const int32_t* Wp = Wt.payload.data();

    if (!Wt.block_scales.empty()) {
        // Per-block weights: scale varies within the tensor, so apply it during
        // accumulation. Each W element W[k*N+j] uses block (k*N+j)/block_size.
        const int bs = Wt.block_size;
        std::vector<double> facc(static_cast<size_t>(N), 0.0);
        for (int k = 0; k < K; ++k) {
            int32_t xv = Xp[k];
            if (xv == 0) continue;
            const double xvd = static_cast<double>(xv);
            const size_t base = static_cast<size_t>(k) * static_cast<size_t>(N);
            const int32_t* w_row = Wp + base;
            for (int j = 0; j < N; ++j) {
                const size_t blk = (base + static_cast<size_t>(j))
                                   / static_cast<size_t>(bs);
                facc[static_cast<size_t>(j)] +=
                    xvd * static_cast<double>(w_row[j])
                        * static_cast<double>(Wt.block_scales[blk]);
            }
        }
        const float xs = Xt.scale;
        for (int j = 0; j < N; ++j)
            out[static_cast<size_t>(j)] =
                static_cast<float>(facc[static_cast<size_t>(j)] * static_cast<double>(xs));
    } else {
        std::vector<int64_t> acc(static_cast<size_t>(N), 0);
        for (int k = 0; k < K; ++k) {
            int32_t xv = Xp[k];
            if (xv == 0) continue;          // sparse trit fast-path
            int64_t xv64 = static_cast<int64_t>(xv);
            const int32_t* w_row = Wp + static_cast<size_t>(k) * static_cast<size_t>(N);
            for (int j = 0; j < N; ++j) {
                acc[static_cast<size_t>(j)] += xv64 * static_cast<int64_t>(w_row[j]);
            }
        }
        float s = Xt.scale * Wt.scale;
        for (int j = 0; j < N; ++j) {
            out[static_cast<size_t>(j)] = static_cast<float>(acc[static_cast<size_t>(j)]) * s;
        }
    }
    // Account TVMACs analytically: kernel-routed path would issue ceil(K/27)
    // tile invocations per output cell.
    sim.counters().bump_n(Opcode::TVMAC,
        static_cast<uint64_t>((K + 26) / 27) * static_cast<uint64_t>(N));
    (void)kt;
}

}  // namespace

void forward_layer(
    Sim& sim,
    KernelTable& kt,
    const LayerWeights& L,
    KVCache& cache,
    const std::vector<float>& hidden_in,
    int pos,
    int hidden_size,
    int head_dim,
    int n_heads,
    int n_kv_heads,
    int intermediate_size,
    float rmsnorm_eps,
    double rope_theta,
    bool ffn_relu2,
    const LutAddrs& luts,
    std::vector<float>& hidden_out,
    BrandonState* state,
    int layer_idx)
{
    KernelId id_mm   = kt.find("tk_matmul_b_9t");
    KernelId id_rms  = kt.find("tk_rmsnorm");
    KernelId id_silu = kt.find("tk_silu");
    KernelId id_sm   = kt.find("tk_softmax");
    KernelId id_rope = kt.find("tk_rope");

    // ---- Attention block ----

    // 1) attn_norm: RMSNorm of hidden_in (kernel for N<=27, host for N>27)
    std::vector<float> x_norm;
    if (hidden_size <= 27)
        rmsnorm_kernel(sim, kt, id_rms, hidden_in, L.attn_norm_w, luts.rsqrt, rmsnorm_eps, x_norm);
    else
        rmsnorm_host(hidden_in, L.attn_norm_w, rmsnorm_eps, x_norm);

    // 2) Q = x_norm @ Wq,  K = x_norm @ Wk,  V = x_norm @ Wv
    std::vector<float> q, k, v;
    const int q_dim  = n_heads    * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    if (g_fp_bypass) {
        mm_row_fp(x_norm, L.Wq, hidden_size, q_dim,  q);
        mm_row_fp(x_norm, L.Wk, hidden_size, kv_dim, k);
        mm_row_fp(x_norm, L.Wv, hidden_size, kv_dim, v);
    } else {
        TritTensor xt = quantize(x_norm.data(), {1, hidden_size}, 9);
        mm_row(sim, kt, id_mm, xt, 0, L.Wq, hidden_size, q_dim,  q);
        mm_row(sim, kt, id_mm, xt, 0, L.Wk, hidden_size, kv_dim, k);
        mm_row(sim, kt, id_mm, xt, 0, L.Wv, hidden_size, kv_dim, v);
    }

    // 2.5) Brandon value_residual hook (§4b of integration guide).
    // V from layer 0 is captured per-token; later layers add v_first to V (no learned alpha).
    if (state && state->use_value_residual) {
        if (layer_idx == 0) {
            if (!state->v_first_captured) {
                state->v_first.assign(v.begin(), v.end());
                state->v_first_captured = true;
            }
        } else {
            for (int i = 0; i < kv_dim; ++i) v[i] += state->v_first[static_cast<size_t>(i)];
        }
    }

    // 3) Apply RoPE to Q and K per head via tk_rope (interleaved layout)
    for (int h = 0; h < n_heads; ++h) {
        std::vector<float> qh(q.begin() + h * head_dim,
                              q.begin() + (h + 1) * head_dim);
        if (head_dim <= 26)  // even pairs only fit cleanly within 27 lanes
            rope_kernel(sim, kt, id_rope, qh, pos, head_dim, rope_theta);
        else
            rope_host(qh, pos, head_dim, rope_theta);
        std::copy(qh.begin(), qh.end(), q.begin() + h * head_dim);
    }
    for (int h = 0; h < n_kv_heads; ++h) {
        std::vector<float> kh(k.begin() + h * head_dim,
                              k.begin() + (h + 1) * head_dim);
        if (head_dim <= 26)
            rope_kernel(sim, kt, id_rope, kh, pos, head_dim, rope_theta);
        else
            rope_host(kh, pos, head_dim, rope_theta);
        std::copy(kh.begin(), kh.end(), k.begin() + h * head_dim);
    }

    // 3.5) Write post-RoPE K and V into the cache at position `pos`.
    for (int i = 0; i < kv_dim; ++i) {
        cache.K[static_cast<size_t>(pos) * kv_dim + i] = k[i];
        cache.V[static_cast<size_t>(pos) * kv_dim + i] = v[i];
    }

    // 4) Causal multi-token attention over cache[0..pos] (inclusive).
    //    For each query head h, compute scores against all cached K positions,
    //    softmax, then weighted sum over cached V (with GQA support).
    const int gqa_group = n_heads / n_kv_heads;   // 1 if not GQA
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> ctx(q_dim, 0.0f);
    for (int h = 0; h < n_heads; ++h) {
        const int kv_h = h / gqa_group;
        const float* qh = q.data() + h * head_dim;

        // scores[t] = Q[h] · K_cache[t][kv_h] / sqrt(head_dim) for t in [0, pos]
        std::vector<float> scores(pos + 1, 0.0f);
        for (int t = 0; t <= pos; ++t) {
            const float* kh = cache.K.data()
                              + static_cast<size_t>(t) * kv_dim
                              + static_cast<size_t>(kv_h) * head_dim;
            double acc = 0.0;
            for (int d = 0; d < head_dim; ++d)
                acc += double(qh[d]) * double(kh[d]);
            scores[t] = static_cast<float>(acc) * inv_sqrt_d;
        }

        // Softmax over scores[0..pos] (kernel for N<=27, host for longer contexts)
        if (static_cast<int>(scores.size()) <= 27)
            softmax_kernel(sim, kt, id_sm, scores, luts.exp, luts.rcp, /*n_pad*/0);
        else
            softmax_host(scores);

        // Weighted sum over V_cache[0..pos][kv_h]
        float* out_h = ctx.data() + h * head_dim;
        for (int t = 0; t <= pos; ++t) {
            const float* vh = cache.V.data()
                              + static_cast<size_t>(t) * kv_dim
                              + static_cast<size_t>(kv_h) * head_dim;
            for (int d = 0; d < head_dim; ++d)
                out_h[d] += scores[t] * vh[d];
        }
    }

    // 5) Wo projection: attn_out = ctx @ Wo
    // BitNet b1.58: apply attn_sub_norm to ctx before Wo projection. The
    // sub-norm gain absorbs the per-tensor weight scale gamma so the {-1,0,+1}
    // matmul produces the correct magnitude.
    if (!L.attn_sub_norm_w.empty()) {
        std::vector<float> ctx_normed;
        rmsnorm_host(ctx, L.attn_sub_norm_w, rmsnorm_eps, ctx_normed);
        ctx = std::move(ctx_normed);
    }
    std::vector<float> attn_out;
    if (g_fp_bypass) {
        mm_row_fp(ctx, L.Wo, q_dim, hidden_size, attn_out);
    } else {
        TritTensor ctxt = quantize(ctx.data(), {1, q_dim}, 9);
        mm_row(sim, kt, id_mm, ctxt, 0, L.Wo, q_dim, hidden_size, attn_out);
    }

    // 6) Residual: hidden_mid = hidden_in + attn_out
    std::vector<float> hidden_mid(static_cast<size_t>(hidden_size));
    for (int i = 0; i < hidden_size; ++i)
        hidden_mid[static_cast<size_t>(i)] = hidden_in[static_cast<size_t>(i)]
                                           + attn_out[static_cast<size_t>(i)];

    // ---- FFN block ----

    // 7) ffn_norm: RMSNorm of hidden_mid (kernel for N<=27, host for N>27)
    std::vector<float> mid_norm;
    if (hidden_size <= 27)
        rmsnorm_kernel(sim, kt, id_rms, hidden_mid, L.ffn_norm_w, luts.rsqrt, rmsnorm_eps, mid_norm);
    else
        rmsnorm_host(hidden_mid, L.ffn_norm_w, rmsnorm_eps, mid_norm);

    // 8) gate = mid_norm @ Wgate;  up = mid_norm @ Wup
    std::vector<float> gate, up;
    if (g_fp_bypass) {
        mm_row_fp(mid_norm, L.Wgate, hidden_size, intermediate_size, gate);
        mm_row_fp(mid_norm, L.Wup,   hidden_size, intermediate_size, up);
    } else {
        TritTensor mt = quantize(mid_norm.data(), {1, hidden_size}, 9);
        mm_row(sim, kt, id_mm, mt, 0, L.Wgate, hidden_size, intermediate_size, gate);
        mm_row(sim, kt, id_mm, mt, 0, L.Wup,   hidden_size, intermediate_size, up);
    }

    // 9) FFN activation: SwiGLU (SiLU·up) for Llama, ReLU²·up for BitNet 2B-4T.
    std::vector<float> ff;
    if (ffn_relu2)
        relu2_mul_host(gate, up, ff);
    else if (intermediate_size <= 27)
        silu_mul_kernel(sim, kt, id_silu, gate, up, luts.sigmoid, ff);
    else
        silu_mul_host(gate, up, ff);

    // 10) Wdown projection: ff_out = ff @ Wdown
    // BitNet b1.58: apply ffn_sub_norm to silu(gate)*up before Wdown.
    if (!L.ffn_sub_norm_w.empty()) {
        std::vector<float> ff_normed;
        rmsnorm_host(ff, L.ffn_sub_norm_w, rmsnorm_eps, ff_normed);
        ff = std::move(ff_normed);
    }
    std::vector<float> ff_out;
    if (g_fp_bypass) {
        mm_row_fp(ff, L.Wdown, intermediate_size, hidden_size, ff_out);
    } else {
        TritTensor fft = quantize(ff.data(), {1, intermediate_size}, 9);
        mm_row(sim, kt, id_mm, fft, 0, L.Wdown, intermediate_size, hidden_size, ff_out);
    }

    // 11) Residual: hidden_out = hidden_mid + ff_out
    hidden_out.assign(static_cast<size_t>(hidden_size), 0.0f);
    for (int i = 0; i < hidden_size; ++i)
        hidden_out[static_cast<size_t>(i)] = hidden_mid[static_cast<size_t>(i)]
                                           + ff_out[static_cast<size_t>(i)];

    // 12) Brandon DWA hook (§4c). After post-FFN hidden state, mix all prior states
    // via the dwa_weights row for this layer. dwa_buf[0] holds the pre-loop input;
    // dwa_buf[(L+1)*H..] holds output of layer L (filled progressively).
    if (state && state->use_dwa && state->dwa_weights) {
        const int H  = hidden_size;
        const int N1 = state->n_layers + 1;
        // Store this layer's output into the buffer slot.
        for (int i = 0; i < H; ++i)
            state->dwa_buf[static_cast<size_t>((layer_idx + 1) * H + i)] = hidden_out[static_cast<size_t>(i)];
        // Mix: x = sum_{j=0..layer_idx+1} w[layer_idx, j] * dwa_buf[j]
        const float* w_row = state->dwa_weights + static_cast<size_t>(layer_idx) * static_cast<size_t>(N1);
        for (int i = 0; i < H; ++i) hidden_out[static_cast<size_t>(i)] = 0.0f;
        for (int j = 0; j <= layer_idx + 1; ++j) {
            float w = w_row[j];
            if (w == 0.0f) continue;
            const float* src = state->dwa_buf.data() + static_cast<size_t>(j * H);
            for (int i = 0; i < H; ++i)
                hidden_out[static_cast<size_t>(i)] += w * src[i];
        }
    }
}

}  // namespace ter::tx
