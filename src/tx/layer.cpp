#include <ter/tx/layer.hpp>

namespace ter::tx {

LayerWeights quantize_layer(
    const float* Wq_data, int Wq_rows, int Wq_cols,
    const float* Wk_data, int Wk_rows, int Wk_cols,
    const float* Wv_data, int Wv_rows, int Wv_cols,
    const float* Wo_data, int Wo_rows, int Wo_cols,
    const float* Wgate_data, int Wgate_rows, int Wgate_cols,
    const float* Wup_data, int Wup_rows, int Wup_cols,
    const float* Wdown_data, int Wdown_rows, int Wdown_cols,
    const float* attn_norm_w, int attn_norm_n,
    const float* ffn_norm_w, int ffn_norm_n)
{
    LayerWeights L;
    L.Wq    = quantize(Wq_data,    {Wq_rows,    Wq_cols},    9);
    L.Wk    = quantize(Wk_data,    {Wk_rows,    Wk_cols},    9);
    L.Wv    = quantize(Wv_data,    {Wv_rows,    Wv_cols},    9);
    L.Wo    = quantize(Wo_data,    {Wo_rows,    Wo_cols},    9);
    L.Wgate = quantize(Wgate_data, {Wgate_rows, Wgate_cols}, 9);
    L.Wup   = quantize(Wup_data,   {Wup_rows,   Wup_cols},   9);
    L.Wdown = quantize(Wdown_data, {Wdown_rows, Wdown_cols}, 9);
    L.attn_norm_w.assign(attn_norm_w, attn_norm_w + attn_norm_n);
    L.ffn_norm_w.assign(ffn_norm_w, ffn_norm_w + ffn_norm_n);
    return L;
}

}  // namespace ter::tx
