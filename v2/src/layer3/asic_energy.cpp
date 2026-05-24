#include <tersim/asic_model.hpp>
#include <ter/isa.hpp>

namespace tersim {

namespace {

// Stillmaker-Baas 2017 scaling factors relative to 22nm. Active energy per
// op drops with node (gate cap shrinks), frequency rises (gate delay
// shrinks), leakage drops more slowly (Dennard broken post-28nm). Numbers
// match the published curves for 22→7 (~2× E, 1.7× f) and 7→3 is a
// documented extrapolation (~1.6× E, 1.3× f) — flagged in the paper.
struct NodeFactors {
    double energy_scale;   // multiplier on pJ-per-op vs 22nm
    double freq_scale;     // multiplier on GHz   vs 22nm
    double area_scale;     // multiplier on mm²/kGE vs 22nm
    double leakage_scale;  // multiplier on leakage W/mm² vs 22nm
};

constexpr NodeFactors node_factors(TechNode n) noexcept {
    switch (n) {
        case TechNode::N22nm: return {1.0,   1.0,   1.0,   1.0};
        case TechNode::N7nm:  return {0.50,  1.70,  0.20,  0.70};
        case TechNode::N3nm:  return {0.31,  2.20,  0.09,  0.50};
    }
    return {1.0, 1.0, 1.0, 1.0};
}

}  // namespace

AsicConfig AsicModel::defaults_for(TechNode node) noexcept {
    AsicConfig c;
    c.node = node;
    const auto f = node_factors(node);

    auto scale_pj = [s = f.energy_scale](double& v) noexcept { v *= s; };
    scale_pj(c.pj_tadd);     scale_pj(c.pj_tsub);     scale_pj(c.pj_tneg);
    scale_pj(c.pj_tabs);     scale_pj(c.pj_tlogic3);  scale_pj(c.pj_tcmp);
    scale_pj(c.pj_tsign);    scale_pj(c.pj_tloadi);   scale_pj(c.pj_tbranch);
    scale_pj(c.pj_tcontrol); scale_pj(c.pj_tvadd);    scale_pj(c.pj_tvsub);
    scale_pj(c.pj_tvneg);    scale_pj(c.pj_tvmul);    scale_pj(c.pj_tvmac);
    scale_pj(c.pj_tvsum);    scale_pj(c.pj_tvmax);    scale_pj(c.pj_tvshuf);
    scale_pj(c.pj_tvbcast);
    // Memory energy scales weakly with node (SRAM bitcell shrinks; DRAM
    // is off-die so largely node-independent). Apply sqrt for SRAM.
    const double mem_scale = (f.energy_scale < 1.0)
                                 ? 0.5 * (1.0 + f.energy_scale)
                                 : 1.0;
    c.pj_l1_access *= mem_scale;
    c.pj_l2_access *= mem_scale;
    // DRAM left unscaled.

    c.freq_ghz *= f.freq_scale;
    c.mm2_per_kge_22nm *= f.area_scale;
    c.leakage_w_per_mm2_22nm *= f.leakage_scale;
    return c;
}

double AsicModel::compute_energy_pJ(const ter::OpCounters& c) const noexcept {
    using ter::Opcode;
    double e = 0.0;
    e += static_cast<double>(c.get(Opcode::TADD))    * cfg_.pj_tadd;
    e += static_cast<double>(c.get(Opcode::TSUB))    * cfg_.pj_tsub;
    e += static_cast<double>(c.get(Opcode::TNEG))    * cfg_.pj_tneg;
    e += static_cast<double>(c.get(Opcode::TABS))    * cfg_.pj_tabs;
    e += static_cast<double>(c.get(Opcode::TAND3))   * cfg_.pj_tlogic3;
    e += static_cast<double>(c.get(Opcode::TOR3))    * cfg_.pj_tlogic3;
    e += static_cast<double>(c.get(Opcode::TXOR3))   * cfg_.pj_tlogic3;
    e += static_cast<double>(c.get(Opcode::TCMP))    * cfg_.pj_tcmp;
    e += static_cast<double>(c.get(Opcode::TSIGN))   * cfg_.pj_tsign;
    e += static_cast<double>(c.get(Opcode::TLOADI))  * cfg_.pj_tloadi;
    e += static_cast<double>(c.get(Opcode::TBEQ))    * cfg_.pj_tbranch;
    e += static_cast<double>(c.get(Opcode::TBNE))    * cfg_.pj_tbranch;
    e += static_cast<double>(c.get(Opcode::TBLT))    * cfg_.pj_tbranch;
    e += static_cast<double>(c.get(Opcode::TJUMP))   * cfg_.pj_tbranch;
    e += static_cast<double>(c.get(Opcode::TCALL))   * cfg_.pj_tbranch;
    e += static_cast<double>(c.get(Opcode::TRET))    * cfg_.pj_tbranch;
    e += static_cast<double>(c.get(Opcode::TNOP))    * cfg_.pj_tcontrol;
    e += static_cast<double>(c.get(Opcode::THALT))   * cfg_.pj_tcontrol;
    e += static_cast<double>(c.get(Opcode::TDBG))    * cfg_.pj_tcontrol;
    e += static_cast<double>(c.get(Opcode::TVADD))   * cfg_.pj_tvadd;
    e += static_cast<double>(c.get(Opcode::TVSUB))   * cfg_.pj_tvsub;
    e += static_cast<double>(c.get(Opcode::TVNEG))   * cfg_.pj_tvneg;
    e += static_cast<double>(c.get(Opcode::TVBROADCAST)) * cfg_.pj_tvbcast;
    e += static_cast<double>(c.get(Opcode::TVMAC))   * cfg_.pj_tvmac;
    e += static_cast<double>(c.get(Opcode::TVSUM))   * cfg_.pj_tvsum;
    e += static_cast<double>(c.get(Opcode::TVMAX))   * cfg_.pj_tvmax;
    e += static_cast<double>(c.get(Opcode::TVSHUF))  * cfg_.pj_tvshuf;
    e += static_cast<double>(c.get(Opcode::TVMUL))   * cfg_.pj_tvmul;
    // TVLOAD / TVSTORE: control side of the access; the data-side energy
    // is accounted for via memory_energy_pJ().
    return e;
}

double AsicModel::memory_energy_pJ(const AsicWorkload& w) const noexcept {
    // L2 accesses ≈ L1D misses. If the caller didn't pass l2_accesses
    // explicitly, derive it from dram_accesses + remainder. For now we
    // trust the caller-provided fields.
    const double l1 = static_cast<double>(w.l1d_accesses)  * cfg_.pj_l1_access;
    const double l2 = static_cast<double>(w.l2_accesses)   * cfg_.pj_l2_access;
    const double dr = static_cast<double>(w.dram_accesses) * cfg_.pj_dram_access;
    return l1 + l2 + dr;
}

double AsicModel::total_area_mm2() const noexcept {
    const double total_kge = cfg_.kge_alu + cfg_.kge_vec_unit
                           + cfg_.kge_l1d + cfg_.kge_l2 + cfg_.kge_ctrl;
    return total_kge * cfg_.mm2_per_kge_22nm;
}

AsicReport AsicModel::project(const AsicWorkload& w) const noexcept {
    AsicReport r;
    if (w.counters == nullptr || w.tokens == 0) return r;

    const double tokens_d = static_cast<double>(w.tokens);

    r.freq_GHz = cfg_.freq_ghz;
    r.area_mm2 = total_area_mm2();

    const double cycles_d  = static_cast<double>(w.cycles_total);
    const double clk_hz    = cfg_.freq_ghz * 1e9;
    r.wall_time_s_per_token = (clk_hz > 0.0)
                                  ? (cycles_d / clk_hz) / tokens_d
                                  : 0.0;

    const double e_compute = compute_energy_pJ(*w.counters);
    const double e_memory  = memory_energy_pJ(w);
    // Leakage: power × wall_time_total. wall_time_total in seconds.
    const double wall_total_s = r.wall_time_s_per_token * tokens_d;
    const double leak_w = r.area_mm2 * cfg_.leakage_w_per_mm2_22nm;
    const double e_leak_pJ = leak_w * wall_total_s * 1e12;

    r.breakdown.compute_pJ = e_compute;
    r.breakdown.memory_pJ  = e_memory;
    r.breakdown.leakage_pJ = e_leak_pJ;

    const double total_pJ = e_compute + e_memory + e_leak_pJ;
    r.total_energy_pJ_per_token = total_pJ / tokens_d;

    // perf/W: tokens/s ÷ Watts. P = E/t = (total_pJ * 1e-12) / wall_total_s.
    if (wall_total_s > 0.0) {
        const double tok_per_s = tokens_d / wall_total_s;
        const double watts     = (total_pJ * 1e-12) / wall_total_s;
        r.perf_per_watt_tokps  = (watts > 0.0) ? tok_per_s / watts : 0.0;
    }
    return r;
}

}  // namespace tersim
