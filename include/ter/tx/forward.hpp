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

// Per-token state for brandon-specific forward modifications.
// Reset at the start of every forward call (NOT slot lifetime — see integration guide §4b).
struct BrandonState {
    // Value residual learning (§4b)
    bool use_value_residual = false;
    bool v_first_captured = false;     // reset to false at the start of each forward call
    std::vector<float> v_first;        // size = n_kv_heads * head_dim, captured at layer 0

    // DenseFormer DWA mixing (§4c)
    bool use_dwa = false;
    int  n_layers = 0;
    int  hidden_size = 0;
    std::vector<float> dwa_buf;        // size = (n_layers + 1) * hidden_size; dwa_buf[(L+1)*H..] holds layer L output
    const float* dwa_weights = nullptr; // (n_layers+1)*n_layers row-major; weights[L*(n_layers+1)+j]
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
    std::vector<float>& hidden_out,
    BrandonState* state = nullptr,           // nullptr = vanilla Llama
    int layer_idx = 0);                       // for v_first capture and DWA buf indexing

}  // namespace ter::tx
