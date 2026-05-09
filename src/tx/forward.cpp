#include <ter/tx/forward.hpp>
#include <ter/numfmt.hpp>
#include <cmath>
#include <algorithm>

namespace ter::tx {

namespace {

// Sim memory map for forward_layer (single-token, single-layer).
// Kernel-code occupies [0, 511]; data starts >= 512.
// We use >= 1024 to be safe.
constexpr int SCRATCH_X = 1024;  // matmul X gather buffer (27 words)
constexpr int SCRATCH_W = 1100;  // matmul W column gather buffer (27 words)
constexpr int Y_TILE    = 1200;  // matmul output tile (1 word)

// Apply RMSNorm: y = (x / rms(x)) * weight, host-only for tractability.
// Future F5.3: route through tk_rmsnorm kernel.
void rmsnorm_host(const std::vector<float>& x, const std::vector<float>& w,
                  float eps, std::vector<float>& y) {
    double ss = 0.0;
    for (auto v : x) ss += double(v) * double(v);
    double rms_inv = 1.0 / std::sqrt(ss / static_cast<double>(x.size()) + eps);
    y.assign(x.size(), 0.0f);
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = static_cast<float>(double(x[i]) * rms_inv) * w[i];
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

// SiLU(gate) * up element-wise.
// Host-only; F5.3 plumbs tk_silu.
void silu_mul_host(const std::vector<float>& gate, const std::vector<float>& up,
                   std::vector<float>& y) {
    y.assign(gate.size(), 0.0f);
    for (size_t i = 0; i < gate.size(); ++i) {
        double s = 1.0 / (1.0 + std::exp(-double(gate[i])));
        y[i] = static_cast<float>(double(gate[i]) * s * double(up[i]));
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
    const std::vector<float>& hidden_in,
    int pos,
    int hidden_size,
    int head_dim,
    int n_heads,
    int n_kv_heads,
    int intermediate_size,
    float rmsnorm_eps,
    const LutAddrs& /*luts*/,  // unused in this MVP — host-side norms/softmax
    std::vector<float>& hidden_out)
{
    KernelId id_mm = kt.find("tk_matmul_b_9t");

    // ---- Attention block ----

    // 1) attn_norm: RMSNorm of hidden_in
    std::vector<float> x_norm;
    rmsnorm_host(hidden_in, L.attn_norm_w, rmsnorm_eps, x_norm);

    // 2) Q = x_norm @ Wq,  K = x_norm @ Wk,  V = x_norm @ Wv
    TritTensor xt = quantize(x_norm.data(), {1, hidden_size}, 9);
    std::vector<float> q, k, v;
    const int q_dim  = n_heads    * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    mm_row(sim, kt, id_mm, xt, 0, L.Wq, hidden_size, q_dim,  q);
    mm_row(sim, kt, id_mm, xt, 0, L.Wk, hidden_size, kv_dim, k);
    mm_row(sim, kt, id_mm, xt, 0, L.Wv, hidden_size, kv_dim, v);

    // 3) Apply RoPE to Q and K per head
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

    // 4) Trivial single-token attention: softmax of one element is 1.0, so
    //    ctx = V directly.  (Multi-token + KV cache lands in F5.3.)
    std::vector<float> ctx = v;  // same shape as v (kv_dim)

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

    // 7) ffn_norm: RMSNorm of hidden_mid
    std::vector<float> mid_norm;
    rmsnorm_host(hidden_mid, L.ffn_norm_w, rmsnorm_eps, mid_norm);

    // 8) gate = mid_norm @ Wgate;  up = mid_norm @ Wup
    TritTensor mt = quantize(mid_norm.data(), {1, hidden_size}, 9);
    std::vector<float> gate, up;
    mm_row(sim, kt, id_mm, mt, 0, L.Wgate, hidden_size, intermediate_size, gate);
    mm_row(sim, kt, id_mm, mt, 0, L.Wup,   hidden_size, intermediate_size, up);

    // 9) SwiGLU: SiLU(gate) * up
    std::vector<float> ff;
    silu_mul_host(gate, up, ff);

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
