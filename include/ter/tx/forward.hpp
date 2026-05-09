#pragma once
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/tx/layer.hpp>
#include <vector>

namespace ter::tx {

// Single-token, single-layer forward pass.
// hidden_in: float vector of length hidden_size (the layer input).
// hidden_out: float vector of length hidden_size (the layer output, post-residual).
// pos: token position (for RoPE).
// head_dim, n_heads, n_kv_heads, intermediate_size: model dims.
// rmsnorm_eps: typically 1e-6.
// LUT addresses must be pre-loaded by the caller (rsqrt, exp, rcp, sigmoid).
// NOTE: luts is currently unused — RMSNorm/RoPE/SiLU/Softmax are host-side for this MVP.
//       Kernel-routed versions land in F5.3.
struct LutAddrs {
    int rsqrt;
    int exp;
    int rcp;
    int sigmoid;
};

void forward_layer(
    Sim& sim,
    KernelTable& kt,
    const LayerWeights& L,
    KVCache& cache,                          // in/out per-layer KV state
    const std::vector<float>& hidden_in,
    int pos,
    int hidden_size,
    int head_dim,
    int n_heads,
    int n_kv_heads,
    int intermediate_size,
    float rmsnorm_eps,
    const LutAddrs& luts,
    std::vector<float>& hidden_out);

}  // namespace ter::tx
