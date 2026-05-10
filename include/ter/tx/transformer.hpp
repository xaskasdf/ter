#pragma once
#include <ter/tx/layer.hpp>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/tx/forward.hpp>
#include <vector>
#include <model/loader.h>

namespace ter::tx {

// A loaded brandon-style Transformer ready for inference.
//
// Holds all weights quantized to Format B (TritFP_B 9-trit), plus the small
// host-side state needed for register prefill / value_residual / DWA mixing.
//
// Memory layout matches the brandon-tiny GGUF: 12 unique blocks fanned out
// to 24 logical layers via cfg.brandon.layer_map.
struct BrandonTransformer {
    // Sized arrays
    std::vector<LayerWeights> blocks;     // size = brandon.block_count (12 for brandon-tiny)
    std::vector<KVCache>      kv_caches;  // size = brandon.compute_layer_count (24); per-logical-layer

    // Quantized embedding/output (weight_tying => same tensor used for both;
    // otherwise lm_head holds the separate output.weight).
    TritTensor token_embd;                // (vocab_size, hidden_size) — F16 source
    TritTensor lm_head;                   // empty when weight_tying=true

    // Norms/aux kept as float (small, no per-tensor scale needed)
    std::vector<float> output_norm_w;     // (hidden_size,)
    std::vector<float> register_w;        // (n_registers, hidden_size)
    std::vector<float> dwa_w;             // (n_layers+1, n_layers) row-major; dwa[L][j] = dwa_w[L*(n_layers+1)+j]
    // NOTE: GGUF stores dwa with shape labels [25,24]; bytes are C-row-major matching the guide's
    // dwa[L][j] = dwa[L*25+j] convention (n_layers+1=25 elements per logical layer L).

    // Cached config (just the values forward_token needs)
    int vocab_size = 0;
    int hidden_size = 0;
    int head_dim = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int intermediate_size = 0;
    int n_layers = 0;          // = brandon.compute_layer_count
    int n_registers = 0;
    bool use_dwa = false;
    bool use_value_residual = false;
    bool weight_tying = false;
    float rmsnorm_eps = 1e-5f;
    std::vector<int> layer_map;   // size = n_layers; values in [0, blocks.size())
};

// Load all weights from a brandon GGUF into a BrandonTransformer.
// Uses tensor_to_trit() for the projection weights; norms / register / dwa stay as float.
// n_trits: Format B trits per element (default 9; bump to 12 for higher fidelity).
// Throws std::runtime_error if a required tensor is missing.
BrandonTransformer load_brandon_transformer(const nt::GGUFLoader& loader,
                                            int max_seq_len,
                                            int n_trits = 9);

// Forward one token through all logical layers; returns logits over vocab_size.
//
// token_id: vocab index, or -1 to use a pre-supplied hidden vector via `hidden_override`.
// pos: absolute position used for RoPE.
// state: per-token brandon mods. Reset/configured by the caller (typically per chat call).
// hidden_override: if non-null, used instead of token_embd[token_id]. Length must be hidden_size.
std::vector<float> forward_token(
    Sim& sim,
    KernelTable& kt,
    BrandonTransformer& tx,
    int token_id,
    int pos,
    const LutAddrs& luts,
    BrandonState* state = nullptr,
    const std::vector<float>* hidden_override = nullptr);

// Load a plain Llama (e.g., Llama 3.2 1B) into the same BrandonTransformer struct
// with identity layer_map and all brandon bits disabled. weight_tying=true: the
// model uses token_embd as the output projection (no separate output.weight).
//
// vocab_size is derived from token_embd shape because llama.* GGUF metadata does
// not include vocab_size. Other fields come from cfg.{hidden_size,n_heads,...}
// which from_gguf_metadata populates from the architecture-prefixed keys.
BrandonTransformer load_llama_transformer(const nt::GGUFLoader& loader,
                                          int max_seq_len,
                                          int n_trits = 9);

// Run the n_registers register-token prefill (per integration guide §4d).
// MUST be called once at the start of every chat session — KV cache for slots 0..n-1
// must hold THIS chat's register activations, not the previous chat's. The caller is
// responsible for resetting tx.kv_caches before this call.
//
// Returns the next position (= n_registers); the user's first token goes at pos = n_registers.
int register_prefill(
    Sim& sim,
    KernelTable& kt,
    BrandonTransformer& tx,
    const LutAddrs& luts,
    BrandonState* state);

}  // namespace ter::tx
