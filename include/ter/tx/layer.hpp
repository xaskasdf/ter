#pragma once
#include <ter/numfmt.hpp>
#include <vector>

namespace ter::tx {

struct LayerWeights {
    TritTensor Wq;
    TritTensor Wk;
    TritTensor Wv;
    TritTensor Wo;
    TritTensor Wgate;
    TritTensor Wup;
    TritTensor Wdown;
    std::vector<float> attn_norm_w;
    std::vector<float> ffn_norm_w;
    // BitNet b1.58 sub-norms: empty for non-BitNet architectures.
    // attn_sub_norm: (hidden,) -- applied to attention output BEFORE Wo projection.
    // ffn_sub_norm:  (intermediate,) -- applied to silu(gate)*up BEFORE Wdown.
    // The sub-norm gain absorbs BitNet's per-tensor weight scale gamma.
    std::vector<float> attn_sub_norm_w;
    std::vector<float> ffn_sub_norm_w;
};

LayerWeights quantize_layer(
    const float* Wq_data, int Wq_rows, int Wq_cols,
    const float* Wk_data, int Wk_rows, int Wk_cols,
    const float* Wv_data, int Wv_rows, int Wv_cols,
    const float* Wo_data, int Wo_rows, int Wo_cols,
    const float* Wgate_data, int Wgate_rows, int Wgate_cols,
    const float* Wup_data, int Wup_rows, int Wup_cols,
    const float* Wdown_data, int Wdown_rows, int Wdown_cols,
    const float* attn_norm_w, int attn_norm_n,
    const float* ffn_norm_w, int ffn_norm_n);

struct KVCache {
    int max_seq = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    std::vector<float> K;   // (max_seq, n_kv_heads * head_dim)
    std::vector<float> V;   // (max_seq, n_kv_heads * head_dim)

    void resize(int new_max_seq, int new_n_kv_heads, int new_head_dim) {
        max_seq = new_max_seq;
        n_kv_heads = new_n_kv_heads;
        head_dim = new_head_dim;
        size_t n = static_cast<size_t>(max_seq) *
                   static_cast<size_t>(n_kv_heads) *
                   static_cast<size_t>(head_dim);
        K.assign(n, 0.0f);
        V.assign(n, 0.0f);
    }

    int kv_dim() const { return n_kv_heads * head_dim; }
};

}  // namespace ter::tx
