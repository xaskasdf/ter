#include <ter/tx/transformer.hpp>
#include <ter/host/load_gguf.hpp>
#include <ter/numfmt.hpp>
#include <core/dequant.hpp>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ter::tx {

namespace {

// Quantize an nt::Tensor to TritTensor via the F5.4a bridge.
TritTensor quant_t(const nt::Tensor& t) {
    return ter::host::tensor_to_trit(t, 9);
}

// Read a float-typed tensor (F32) into a flat vector. The brandon norm/dwa weights
// are F32 per the integration guide.
std::vector<float> as_floats(const nt::Tensor& t) {
    size_t n = 1;
    for (auto d : t.shape()) n *= static_cast<size_t>(d);
    std::vector<float> out(n);
    if (t.dtype() == nt::DType::F32) {
        nt::dequant_f32(t.data(), n, out.data());
    } else if (t.dtype() == nt::DType::F16) {
        nt::dequant_f16(t.data(), n, out.data());
    } else {
        throw std::runtime_error("as_floats: only F16/F32 supported");
    }
    return out;
}

// Build one block's LayerWeights from the loader's named tensors.
LayerWeights build_block(const nt::GGUFLoader& loader, int b) {
    auto pfx = std::string("blk.") + std::to_string(b) + ".";
    auto Wq        = quant_t(loader.get_tensor(pfx + "attn_q.weight"));
    auto Wk        = quant_t(loader.get_tensor(pfx + "attn_k.weight"));
    auto Wv        = quant_t(loader.get_tensor(pfx + "attn_v.weight"));
    auto Wo        = quant_t(loader.get_tensor(pfx + "attn_output.weight"));
    auto Wgate     = quant_t(loader.get_tensor(pfx + "ffn_gate.weight"));
    auto Wup       = quant_t(loader.get_tensor(pfx + "ffn_up.weight"));
    auto Wdown     = quant_t(loader.get_tensor(pfx + "ffn_down.weight"));
    auto attn_norm = as_floats(loader.get_tensor(pfx + "attn_norm.weight"));
    auto ffn_norm  = as_floats(loader.get_tensor(pfx + "ffn_norm.weight"));

    LayerWeights L;
    L.Wq = std::move(Wq);
    L.Wk = std::move(Wk);
    L.Wv = std::move(Wv);
    L.Wo = std::move(Wo);
    L.Wgate = std::move(Wgate);
    L.Wup   = std::move(Wup);
    L.Wdown = std::move(Wdown);
    L.attn_norm_w = std::move(attn_norm);
    L.ffn_norm_w  = std::move(ffn_norm);
    return L;
}

}  // namespace

BrandonTransformer load_brandon_transformer(const nt::GGUFLoader& loader, int max_seq_len) {
    BrandonTransformer tx;
    const auto& cfg = loader.config();
    const auto& b   = cfg.brandon;

    tx.vocab_size        = cfg.vocab_size;
    tx.hidden_size       = cfg.hidden_size;
    tx.head_dim          = cfg.head_dim;
    tx.n_heads           = cfg.n_heads;
    tx.n_kv_heads        = cfg.n_kv_heads;
    tx.intermediate_size = cfg.intermediate_size;
    tx.n_layers          = cfg.n_layers;        // == compute_layer_count for brandon
    tx.rmsnorm_eps       = cfg.norm_eps;
    tx.n_registers       = b.n_registers;
    tx.use_dwa           = b.use_dwa;
    tx.use_value_residual = b.use_value_residual;
    tx.weight_tying      = b.weight_tying;
    tx.layer_map         = b.layer_map;

    // Load the 12 unique blocks.
    tx.blocks.reserve(static_cast<size_t>(b.block_count));
    for (int i = 0; i < b.block_count; ++i) {
        tx.blocks.push_back(build_block(loader, i));
    }

    // Load globals.
    tx.token_embd     = quant_t(loader.get_tensor("token_embd.weight"));
    tx.output_norm_w  = as_floats(loader.get_tensor("output_norm.weight"));
    if (b.n_registers > 0) {
        tx.register_w = as_floats(loader.get_tensor("register.weight"));
    }
    if (b.use_dwa) {
        tx.dwa_w = as_floats(loader.get_tensor("dwa.weight"));
    }

    // KV caches: one per logical layer.
    tx.kv_caches.resize(static_cast<size_t>(tx.n_layers));
    for (auto& c : tx.kv_caches) c.resize(max_seq_len, tx.n_kv_heads, tx.head_dim);

    return tx;
}

std::vector<float> forward_token(
    Sim& sim,
    KernelTable& kt,
    BrandonTransformer& tx,
    int token_id,
    int pos,
    const LutAddrs& luts,
    BrandonState* state,
    const std::vector<float>* hidden_override)
{
    const int H = tx.hidden_size;
    std::vector<float> hidden(static_cast<size_t>(H), 0.0f);
    std::vector<float> emb_full;

    if (hidden_override) {
        if (static_cast<int>(hidden_override->size()) != H) {
            throw std::runtime_error("forward_token: hidden_override size != hidden_size");
        }
        hidden = *hidden_override;
    } else {
        if (token_id < 0 || token_id >= tx.vocab_size) {
            throw std::runtime_error("forward_token: token_id out of range");
        }
        // Token embedding lookup. token_embd payload is row-major (vocab_size rows, H cols).
        emb_full.resize(tx.token_embd.payload.size());
        ter::dequantize(tx.token_embd, emb_full.data());
        for (int i = 0; i < H; ++i) {
            hidden[static_cast<size_t>(i)] =
                emb_full[static_cast<size_t>(token_id) * static_cast<size_t>(H) + static_cast<size_t>(i)];
        }
    }

    // Brandon DWA: capture the pre-loop hidden vector into dwa_buf[0..H).
    if (state && state->use_dwa) {
        if (static_cast<int>(state->dwa_buf.size()) != (state->n_layers + 1) * H) {
            throw std::runtime_error("forward_token: dwa_buf wrong size");
        }
        for (int i = 0; i < H; ++i) state->dwa_buf[static_cast<size_t>(i)] = hidden[static_cast<size_t>(i)];
    }
    // Brandon value_residual: reset capture flag at the start of every forward call (§4b).
    if (state && state->use_value_residual) state->v_first_captured = false;

    // 24 logical layers via layer_map fanout.
    std::vector<float> hidden_out(static_cast<size_t>(H));
    for (int L = 0; L < tx.n_layers; ++L) {
        int b_idx = tx.layer_map[static_cast<size_t>(L)];
        if (b_idx < 0 || b_idx >= static_cast<int>(tx.blocks.size())) {
            throw std::runtime_error("forward_token: layer_map index out of range");
        }
        forward_layer(
            sim, kt, tx.blocks[static_cast<size_t>(b_idx)], tx.kv_caches[static_cast<size_t>(L)],
            hidden, pos,
            tx.hidden_size, tx.head_dim, tx.n_heads, tx.n_kv_heads,
            tx.intermediate_size, tx.rmsnorm_eps, luts,
            hidden_out, state, L);
        hidden = hidden_out;
    }

    // Register prefill case: hidden_override was used and we don't need logits.
    // Skip output norm + projection entirely.
    if (hidden_override) {
        return {};
    }

    // 3) Output norm (RMSNorm with output_norm_w gain) — host-side (H=256 > 27).
    {
        double ss = 0.0;
        for (auto v : hidden) ss += double(v) * double(v);
        double rms_inv = 1.0 / std::sqrt(ss / static_cast<double>(H) + double(tx.rmsnorm_eps));
        for (size_t i = 0; i < hidden.size(); ++i)
            hidden[i] = static_cast<float>(static_cast<double>(hidden[i]) * rms_inv) * tx.output_norm_w[i];
    }

    // 4) Logits = hidden @ token_embd.T  (with weight_tying, output uses the same matrix).
    // token_embd payload is (vocab_size, hidden_size) row-major after the GGUF reversal trick.
    // logits[v] = sum_i hidden[i] * embd[v, i]
    std::vector<float> logits(static_cast<size_t>(tx.vocab_size), 0.0f);
    for (int v = 0; v < tx.vocab_size; ++v) {
        double acc = 0.0;
        for (int i = 0; i < H; ++i) {
            acc += static_cast<double>(hidden[static_cast<size_t>(i)])
                 * static_cast<double>(emb_full[static_cast<size_t>(v) * static_cast<size_t>(H)
                                                + static_cast<size_t>(i)]);
        }
        logits[static_cast<size_t>(v)] = static_cast<float>(acc);
    }
    return logits;
}

int register_prefill(
    Sim& sim,
    KernelTable& kt,
    BrandonTransformer& tx,
    const LutAddrs& luts,
    BrandonState* state)
{
    if (tx.n_registers <= 0) return 0;
    const int H = tx.hidden_size;
    if (static_cast<int>(tx.register_w.size()) != tx.n_registers * H) {
        throw std::runtime_error("register_prefill: register_w wrong size");
    }
    for (int r = 0; r < tx.n_registers; ++r) {
        std::vector<float> reg_emb(static_cast<size_t>(H));
        // register_w stored row-major: row r is registers[r * H .. (r+1) * H).
        for (int i = 0; i < H; ++i)
            reg_emb[static_cast<size_t>(i)] = tx.register_w[static_cast<size_t>(r) * static_cast<size_t>(H)
                                                         + static_cast<size_t>(i)];
        // Run forward at position r without producing logits.
        forward_token(sim, kt, tx, /*token_id=*/-1, /*pos=*/r, luts, state, &reg_emb);
    }
    return tx.n_registers;
}

}  // namespace ter::tx
