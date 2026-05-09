#pragma once

#include "../core/types.h"
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace nt {

// ============================================================
// Model configuration parsed from GGUF metadata
// Supports Llama, Mistral, and similar architectures
// ============================================================

struct BrandonConfig {
    int   block_count = 0;
    int   compute_layer_count = 0;
    std::vector<int> layer_map;
    int   n_registers = 0;
    int   n_loops = 1;
    bool  use_dwa = false;
    bool  use_value_residual = false;
    bool  weight_tying = false;

    bool is_valid() const {
        return block_count > 0 && static_cast<int>(layer_map.size()) == compute_layer_count;
    }
};

struct ModelConfig {
    // Architecture
    std::string architecture = "llama";
    std::string model_name;

    // Dimensions
    int vocab_size    = 32000;
    int hidden_size   = 4096;    // d_model
    int intermediate_size = 11008;  // FFN intermediate
    int n_layers      = 32;
    int n_heads       = 32;      // query heads
    int n_kv_heads    = 32;      // key/value heads (GQA if < n_heads)
    int head_dim      = 128;     // hidden_size / n_heads

    // Normalization
    float norm_eps    = 1e-5f;

    // RoPE
    float rope_theta  = 10000.0f;
    float rope_freq_scale = 1.0f;
    bool  rope_interleaved = false;

    // Context
    int max_seq_len   = 4096;

    // Quantization (per model, individual tensors may differ)
    DType weight_dtype = DType::Q4_0;

    // BOS/EOS tokens
    int bos_token_id  = 1;
    int eos_token_id  = 2;

    // Computed values
    int group_size() const { return n_heads / n_kv_heads; }
    bool is_gqa() const { return n_kv_heads < n_heads; }

    // Raw metadata storage
    std::unordered_map<std::string, std::variant<int, float, std::string, bool>> metadata;

    void print() const;

    // Parse from GGUF metadata key-value pairs.
    // brandon_layer_map is populated by the loader when it encounters the
    // brandon.layer_map INT32 array (arrays cannot be stored in the scalar kv map).
    void from_gguf_metadata(
        const std::unordered_map<std::string, std::variant<int, float, std::string, bool>>& kv,
        std::vector<int> brandon_layer_map = {});

    BrandonConfig brandon;
};

} // namespace nt
