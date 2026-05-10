#pragma once
#include <ter/sim.hpp>
#include <ter/isa.hpp>
#include <cstdio>
#include <cstdint>

namespace ter::tx {

// One TVMAC processes a 27-lane vector multiply-accumulate. For the apples-to-
// apples comparison against a binary fp16 implementation, treat each lane of a
// TVMAC as one "ternary MAC" (9-trit * 9-trit + accumulate). The remaining
// scalar/vector ops cost an order of magnitude less than the matmul fabric, so
// the ratio is dominated by tvmac_lane_macs vs analytical fp16 mac count.
struct OpStats {
    uint64_t total_ops = 0;
    uint64_t tvmac     = 0;
    uint64_t tvadd     = 0;
    uint64_t tvsub     = 0;
    uint64_t tvmul     = 0;
    uint64_t tvsum     = 0;
    uint64_t tvbcast   = 0;
    uint64_t tvload    = 0;
    uint64_t tvstore   = 0;
    uint64_t tload     = 0;
    uint64_t tstore    = 0;
    uint64_t tadd      = 0;
    uint64_t tsub      = 0;
    uint64_t tcmp      = 0;
    uint64_t tjump     = 0;
    uint64_t tcall     = 0;
    uint64_t tbranch   = 0;  // TBEQ + TBNE + TBLT

    static OpStats from(const Sim& s) {
        const auto& c = s.counters();
        OpStats x;
        x.total_ops = c.total();
        x.tvmac   = c.get(Opcode::TVMAC);
        x.tvadd   = c.get(Opcode::TVADD);
        x.tvsub   = c.get(Opcode::TVSUB);
        x.tvmul   = c.get(Opcode::TVMUL);
        x.tvsum   = c.get(Opcode::TVSUM);
        x.tvbcast = c.get(Opcode::TVBROADCAST);
        x.tvload  = c.get(Opcode::TVLOAD);
        x.tvstore = c.get(Opcode::TVSTORE);
        x.tload   = c.get(Opcode::TLOAD);
        x.tstore  = c.get(Opcode::TSTORE);
        x.tadd    = c.get(Opcode::TADD);
        x.tsub    = c.get(Opcode::TSUB);
        x.tcmp    = c.get(Opcode::TCMP);
        x.tjump   = c.get(Opcode::TJUMP);
        x.tcall   = c.get(Opcode::TCALL);
        x.tbranch = c.get(Opcode::TBEQ) + c.get(Opcode::TBNE) + c.get(Opcode::TBLT);
        return x;
    }

    // 9-trit lane MACs = TVMAC * 27 lanes.
    uint64_t lane_macs() const noexcept { return tvmac * 27ull; }
};

// Analytical fp16 MAC count for one transformer forward pass at a single token.
// Counts only the heavy projections (Q, K, V, O, gate, up, down, lm_head). Norm
// and softmax are negligible. Attention scoring (Q @ K^T and AV) at 1 token is
// also negligible compared to projections, so we skip it for the headline number.
// Inputs match a standard Llama-style transformer.
inline uint64_t analytical_fp16_macs_per_forward(
    int hidden_size, int intermediate_size,
    int n_heads, int n_kv_heads, int head_dim,
    int n_layers, int vocab_size)
{
    // attn projections at 1 token: Q is H -> n_heads*head_dim, K/V are H -> n_kv*head_dim
    uint64_t H   = static_cast<uint64_t>(hidden_size);
    uint64_t F   = static_cast<uint64_t>(intermediate_size);
    uint64_t HQ  = static_cast<uint64_t>(n_heads)    * static_cast<uint64_t>(head_dim);
    uint64_t HKV = static_cast<uint64_t>(n_kv_heads) * static_cast<uint64_t>(head_dim);
    uint64_t per_layer = H * HQ           // Q
                       + H * HKV          // K
                       + H * HKV          // V
                       + HQ * H           // O
                       + H * F            // gate
                       + H * F            // up
                       + F * H;           // down
    uint64_t lm_head = H * static_cast<uint64_t>(vocab_size);
    return per_layer * static_cast<uint64_t>(n_layers) + lm_head;
}

inline void dump_op_stats(const Sim& s, const char* label,
                          int hidden_size, int intermediate_size,
                          int n_heads, int n_kv_heads, int head_dim,
                          int n_layers, int vocab_size)
{
    OpStats x = OpStats::from(s);
    uint64_t fp16_macs = analytical_fp16_macs_per_forward(
        hidden_size, intermediate_size, n_heads, n_kv_heads, head_dim,
        n_layers, vocab_size);
    uint64_t lane_macs = x.lane_macs();
    double ratio = fp16_macs == 0 ? 0.0
                 : static_cast<double>(lane_macs) / static_cast<double>(fp16_macs);

    std::fprintf(stderr,
        "\n=== op-stats: %s ===\n"
        "  total kernel ops : %llu\n"
        "  TVMAC            : %llu  (lane-MACs: %llu = TVMAC*27)\n"
        "  TVADD/TVSUB/TVMUL: %llu / %llu / %llu\n"
        "  TVSUM/TVBCAST    : %llu / %llu\n"
        "  TVLOAD/TVSTORE   : %llu / %llu\n"
        "  TLOAD/TSTORE     : %llu / %llu\n"
        "  TADD/TSUB/TCMP   : %llu / %llu / %llu\n"
        "  TJUMP/TCALL/Tbr  : %llu / %llu / %llu\n"
        "  -- analytical baseline --\n"
        "  fp16 MACs/forward: %llu  (Q+K+V+O+gate+up+down per layer, +lm_head)\n"
        "  lane-MAC : fp16  : %.3f x  (>1 means ternary substrate does more MACs)\n",
        label,
        (unsigned long long)x.total_ops,
        (unsigned long long)x.tvmac, (unsigned long long)lane_macs,
        (unsigned long long)x.tvadd, (unsigned long long)x.tvsub, (unsigned long long)x.tvmul,
        (unsigned long long)x.tvsum, (unsigned long long)x.tvbcast,
        (unsigned long long)x.tvload, (unsigned long long)x.tvstore,
        (unsigned long long)x.tload, (unsigned long long)x.tstore,
        (unsigned long long)x.tadd, (unsigned long long)x.tsub, (unsigned long long)x.tcmp,
        (unsigned long long)x.tjump, (unsigned long long)x.tcall, (unsigned long long)x.tbranch,
        (unsigned long long)fp16_macs,
        ratio);
}

}  // namespace ter::tx
