#include <ter/tx/forward.hpp>
#include <ter/numfmt.hpp>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace ter::tx {

namespace {

// Sim memory map for forward_layer (single-token, single-layer).
// Kernel-code occupies [0, 511]; data starts >= 512.
// We use >= 1024 to be safe.
constexpr int SCRATCH_X  = 1024;  // matmul X gather buffer (27 words)
constexpr int SCRATCH_W  = 1100;  // matmul W column gather buffer (27 words)
constexpr int Y_TILE     = 1200;  // matmul output tile (1 word)
constexpr int RMS_X_ADDR = 1300;  // rmsnorm input buffer (27 words)
constexpr int RMS_Y_ADDR = 1400;  // rmsnorm output buffer (27 words)

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
        sim.mem().store_word(static_cast<size_t>(RMS_X_ADDR + i), xt.payload[i]);

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

// Apply RoPE to a head-dim vector (single token at position pos).
// Host-only for this MVP; F5.3 plumbs tk_rope.
void rope_host(std::vector<float>& v, int pos, int head_dim) {
    int n_pairs = head_dim / 2;
    for (int k = 0; k < n_pairs; ++k) {
        double freq  = 1.0 / std::pow(10000.0, (2.0 * k) / double(head_dim));
        double angle = double(pos) * freq;
        double c = std::cos(angle), s = std::sin(angle);
        float v0 = v[k];
        float v1 = v[k + n_pairs];
        v[k]          = static_cast<float>(double(v0) * c - double(v1) * s);
        v[k + n_pairs] = static_cast<float>(double(v0) * s + double(v1) * c);
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
        sim.mem().store_word(static_cast<size_t>(x_addr + i), gt.payload[i]);

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

// mm_row: dot row[row] of Xt by all columns of Wt using tk_matmul_b_9t.
// Xt is shape (1, K); Wt is shape (K, N) in row-major order.
// Result: N float values written to `out`.
void mm_row(Sim& sim, KernelTable& kt, KernelId id_mm,
            const TritTensor& Xt, int row,
            const TritTensor& Wt,
            int K, int N,
            std::vector<float>& out) {
    out.assign(static_cast<size_t>(N), 0.0f);
    for (int j = 0; j < N; ++j) {
        int64_t int_acc = 0;
        for (int k0 = 0; k0 < K; k0 += 27) {
            int chunk = std::min(27, K - k0);
            for (int t = 0; t < 27; ++t) {
                int xv = (t < chunk) ? Xt.payload[static_cast<size_t>(row * K + (k0 + t))].to_int() : 0;
                int wv = (t < chunk) ? Wt.payload[static_cast<size_t>((k0 + t) * N + j)].to_int() : 0;
                sim.mem().store_word(static_cast<size_t>(SCRATCH_X + t), Word27::from_int(xv));
                sim.mem().store_word(static_cast<size_t>(SCRATCH_W + t), Word27::from_int(wv));
            }
            std::vector<int64_t> args = {SCRATCH_X, SCRATCH_W, Y_TILE, 0, 0, 0, 0};
            sim.call_kernel(kt, id_mm, args);
            int_acc += sim.mem().load_word(static_cast<size_t>(Y_TILE)).to_int();
        }
        out[static_cast<size_t>(j)] = static_cast<float>(int_acc) * Xt.scale * Wt.scale;
    }
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
    const LutAddrs& luts,
    std::vector<float>& hidden_out)
{
    KernelId id_mm   = kt.find("tk_matmul_b_9t");
    KernelId id_rms  = kt.find("tk_rmsnorm");
    KernelId id_silu = kt.find("tk_silu");

    // ---- Attention block ----

    // 1) attn_norm: RMSNorm of hidden_in via tk_rmsnorm kernel
    std::vector<float> x_norm;
    rmsnorm_kernel(sim, kt, id_rms, hidden_in, L.attn_norm_w, luts.rsqrt, rmsnorm_eps, x_norm);

    // 2) Q = x_norm @ Wq,  K = x_norm @ Wk,  V = x_norm @ Wv
    TritTensor xt = quantize(x_norm.data(), {1, hidden_size}, 9);
    std::vector<float> q, k, v;
    const int q_dim  = n_heads    * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    mm_row(sim, kt, id_mm, xt, 0, L.Wq, hidden_size, q_dim,  q);
    mm_row(sim, kt, id_mm, xt, 0, L.Wk, hidden_size, kv_dim, k);
    mm_row(sim, kt, id_mm, xt, 0, L.Wv, hidden_size, kv_dim, v);

    // 3) Apply RoPE to Q and K per head (before caching K — per Llama spec)
    for (int h = 0; h < n_heads; ++h) {
        std::vector<float> qh(q.begin() + h * head_dim,
                              q.begin() + (h + 1) * head_dim);
        rope_host(qh, pos, head_dim);
        std::copy(qh.begin(), qh.end(), q.begin() + h * head_dim);
    }
    for (int h = 0; h < n_kv_heads; ++h) {
        std::vector<float> kh(k.begin() + h * head_dim,
                              k.begin() + (h + 1) * head_dim);
        rope_host(kh, pos, head_dim);
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

        // Softmax over scores[0..pos]
        float mx = *std::max_element(scores.begin(), scores.end());
        double sum = 0.0;
        for (auto& s : scores) { s = std::exp(s - mx); sum += s; }
        for (auto& s : scores) s = static_cast<float>(s / sum);

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
    TritTensor ctxt = quantize(ctx.data(), {1, q_dim}, 9);
    std::vector<float> attn_out;
    mm_row(sim, kt, id_mm, ctxt, 0, L.Wo, q_dim, hidden_size, attn_out);

    // 6) Residual: hidden_mid = hidden_in + attn_out
    std::vector<float> hidden_mid(static_cast<size_t>(hidden_size));
    for (int i = 0; i < hidden_size; ++i)
        hidden_mid[static_cast<size_t>(i)] = hidden_in[static_cast<size_t>(i)]
                                           + attn_out[static_cast<size_t>(i)];

    // ---- FFN block ----

    // 7) ffn_norm: RMSNorm of hidden_mid via tk_rmsnorm kernel
    std::vector<float> mid_norm;
    rmsnorm_kernel(sim, kt, id_rms, hidden_mid, L.ffn_norm_w, luts.rsqrt, rmsnorm_eps, mid_norm);

    // 8) gate = mid_norm @ Wgate;  up = mid_norm @ Wup
    TritTensor mt = quantize(mid_norm.data(), {1, hidden_size}, 9);
    std::vector<float> gate, up;
    mm_row(sim, kt, id_mm, mt, 0, L.Wgate, hidden_size, intermediate_size, gate);
    mm_row(sim, kt, id_mm, mt, 0, L.Wup,   hidden_size, intermediate_size, up);

    // 9) SwiGLU: SiLU(gate) * up via tk_silu kernel
    std::vector<float> ff;
    silu_mul_kernel(sim, kt, id_silu, gate, up, luts.sigmoid, ff);

    // 10) Wdown projection: ff_out = ff @ Wdown
    TritTensor fft = quantize(ff.data(), {1, intermediate_size}, 9);
    std::vector<float> ff_out;
    mm_row(sim, kt, id_mm, fft, 0, L.Wdown, intermediate_size, hidden_size, ff_out);

    // 11) Residual: hidden_out = hidden_mid + ff_out
    hidden_out.assign(static_cast<size_t>(hidden_size), 0.0f);
    for (int i = 0; i < hidden_size; ++i)
        hidden_out[static_cast<size_t>(i)] = hidden_mid[static_cast<size_t>(i)]
                                           + ff_out[static_cast<size_t>(i)];
}

}  // namespace ter::tx
