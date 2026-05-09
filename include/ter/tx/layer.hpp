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

}  // namespace ter::tx
