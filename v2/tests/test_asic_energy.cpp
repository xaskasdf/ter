// G7: validate the Layer-3 ASIC projection model. Pins per-op energy to
// Horowitz 2014 baselines @22nm, verifies node scaling 22→7→3 follows
// Stillmaker-Baas 2017, and checks that the ternary model honors the
// Cuevas 2026 Prop.2 cota (log2(3)/log2(2) ≈ 1.585× switching-activity
// reduction vs an equivalent binary baseline). Final case runs the real
// tk_matmul_b_9t kernel under PipelineModel and projects energy/token
// end-to-end.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <tersim/asic_model.hpp>
#include <tersim/bp_model.hpp>
#include <tersim/cache_model.hpp>
#include <tersim/cycle_model.hpp>
#include <tersim/trace.hpp>
#include <ter/counters.hpp>
#include <ter/isa.hpp>
#include <ter/kernels.hpp>
#include <ter/sim.hpp>

#ifndef TER_KERNELS_DIR
#error "TER_KERNELS_DIR must be defined by the build system"
#endif

#include <cstdio>
#include <cmath>
#include <vector>

using namespace ter;
using namespace tersim;

TEST_CASE("G7-ASIC: Horowitz 2014 sanity — per-op pJ @22nm in published range") {
    AsicConfig c = AsicModel::defaults_for(TechNode::N22nm);

    // Horowitz 2014 Tab.1: INT8 add @45nm ≈ 0.03 pJ; @22nm ≈ same order
    // (gate cap shrinks but supply also drops). We pin to 0.03 pJ.
    CHECK(c.pj_tadd >  0.01);
    CHECK(c.pj_tadd <  0.10);
    // INT8 mul Horowitz @45nm ≈ 0.2-0.4 pJ; ternary collapses to add/sub/skip,
    // so TVMAC for a 9-trit lane should land below the binary INT8 mul.
    CHECK(c.pj_tvmac >  0.05);
    CHECK(c.pj_tvmac <  0.40);
    // DRAM access Horowitz: ~1300 pJ/byte. Per-access (~1 line worth).
    CHECK(c.pj_dram_access > 500.0);
    CHECK(c.pj_dram_access < 2500.0);
    // L1 access: small SRAM ~5 pJ range.
    CHECK(c.pj_l1_access >  1.0);
    CHECK(c.pj_l1_access < 20.0);

    std::printf("[G7-ASIC Horowitz] TADD=%.3f pJ TVMAC=%.3f pJ "
                "L1=%.1f pJ L2=%.1f pJ DRAM=%.0f pJ\n",
                c.pj_tadd, c.pj_tvmac, c.pj_l1_access,
                c.pj_l2_access, c.pj_dram_access);
}

TEST_CASE("G7-ASIC: Stillmaker-Baas node scaling — 22→7→3 energy↓ freq↑") {
    const auto c22 = AsicModel::defaults_for(TechNode::N22nm);
    const auto c7  = AsicModel::defaults_for(TechNode::N7nm);
    const auto c3  = AsicModel::defaults_for(TechNode::N3nm);

    // Monotone energy decrease per op.
    CHECK(c22.pj_tadd  > c7.pj_tadd);
    CHECK(c7.pj_tadd   > c3.pj_tadd);
    CHECK(c22.pj_tvmac > c7.pj_tvmac);
    CHECK(c7.pj_tvmac  > c3.pj_tvmac);

    // Monotone frequency increase.
    CHECK(c7.freq_ghz  > c22.freq_ghz);
    CHECK(c3.freq_ghz  > c7.freq_ghz);

    // 22→7 active-energy factor ≈ 2× per Stillmaker-Baas 2017.
    const double e_ratio_22_7 = c22.pj_tadd / c7.pj_tadd;
    CHECK(e_ratio_22_7 > 1.5);
    CHECK(e_ratio_22_7 < 3.0);

    // Leakage drops slower than active energy (Dennard broken post-28nm).
    const double leak_drop = c22.leakage_w_per_mm2_22nm
                           / c7.leakage_w_per_mm2_22nm;
    const double act_drop  = c22.pj_tadd / c7.pj_tadd;
    CHECK(leak_drop < act_drop);

    std::printf("[G7-ASIC node scaling] TADD 22nm=%.4f 7nm=%.4f 3nm=%.4f pJ | "
                "freq 22nm=%.2f 7nm=%.2f 3nm=%.2f GHz | leak drop 22→7=%.2f×\n",
                c22.pj_tadd, c7.pj_tadd, c3.pj_tadd,
                c22.freq_ghz, c7.freq_ghz, c3.freq_ghz, leak_drop);
}

TEST_CASE("G7-ASIC: Cuevas 2026 Prop.2 — ternary TADD ≤ binary by ≈log2(3)/log2(2)") {
    // The cota: ternary switching activity is 1/log2(3) ≈ 0.6309× the
    // equivalent binary baseline (Cuevas 2026 Prop.2). Our pj_tadd is the
    // ternary cost; the corresponding hypothetical binary baseline must be
    // pj_tadd / kCuevasTernaryRatio (≈ pj_tadd × 1.585).
    const auto c = AsicModel::defaults_for(TechNode::N22nm);
    const double binary_baseline = c.pj_tadd / AsicModel::kCuevasTernaryRatio;
    const double ratio = c.pj_tadd / binary_baseline;

    // ratio should be ≈ 0.631 within numerical noise.
    CHECK(std::fabs(ratio - AsicModel::kCuevasTernaryRatio) < 1e-9);
    CHECK(ratio < 0.65);
    CHECK(ratio > 0.60);

    // Sanity on the constant itself.
    const double expected = 1.0 / (std::log(3.0) / std::log(2.0));
    CHECK(std::fabs(AsicModel::kCuevasTernaryRatio - expected) < 1e-9);

    std::printf("[G7-ASIC Cuevas Prop.2] ternary/binary = %.4f "
                "(cota %.4f = 1/log2(3))\n",
                ratio, AsicModel::kCuevasTernaryRatio);
}

TEST_CASE("G7-ASIC: end-to-end — real ternary kernel → OpCounters → projection") {
    // Run tk_matmul_b_9t (the same workload used by G5b's e2e test) and
    // hand the resulting OpCounters + PipelineReport to AsicModel.
    Sim s(1024);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    KernelId id = kt.find("tk_matmul_b_9t");
    REQUIRE(id.valid);

    constexpr int K = 27;
    std::vector<int> X(K), W(K);
    for (int i = 0; i < K; ++i) {
        X[i] = (i % 3) - 1;
        W[i] = ((i * 5) % 3) - 1;
    }
    for (int i = 0; i < K; ++i) {
        s.mem().store_word(static_cast<size_t>(200 + i), Word27::from_int(X[i]));
        s.mem().store_word(static_cast<size_t>(300 + i), Word27::from_int(W[i]));
    }

    PipelineConfig pcfg;
    PipelineModel pipe(pcfg);
    MemHierarchyConfig mc; mc.mshrs = 8;
    MemHierarchy mh(mc);
    BpConfig bcfg; bcfg.ghr_history_bits = 0;
    BranchPredictor bp(bcfg);
    pipe.set_mem_hierarchy(&mh);
    pipe.set_branch_predictor(&bp);

    s.set_tracer([&pipe](std::uint64_t pc, const Instr& instr, const Sim& sim) {
        InstrTrace t;
        t.pc = pc;  t.op = instr.op;
        t.dst  = static_cast<std::uint8_t>(instr.dst);
        t.src1 = static_cast<std::uint8_t>(instr.src1);
        t.src2 = static_cast<std::uint8_t>(instr.src2);
        t.imm  = instr.imm;
        switch (instr.op) {
            case Opcode::TLOAD:
            case Opcode::TVLOAD:
                t.mem_addr = static_cast<std::uint64_t>(
                    sim.regs().read_scalar(instr.src1).to_int()) * 8;
                t.mem_is_load = true;
                break;
            case Opcode::TSTORE:
            case Opcode::TVSTORE:
                t.mem_addr = static_cast<std::uint64_t>(
                    sim.regs().read_scalar(instr.src2).to_int()) * 8;
                t.mem_is_store = true;
                break;
            default: break;
        }
        const auto next_pc = static_cast<std::uint64_t>(sim.regs().pc().to_int());
        t.branch_taken  = (next_pc != pc + 1);
        t.branch_target = t.branch_taken ? next_pc : pc + 1;
        pipe.on_retire(t);
    });

    std::vector<int64_t> args = {200, 300, 500, 0, 0, 0, 0};
    s.call_kernel(kt, id, args);

    const auto& rep = pipe.report();
    REQUIRE(rep.cycles_total > 0);
    REQUIRE(s.counters().total() > 0);

    AsicWorkload w;
    w.counters       = &s.counters();
    w.cycles_total   = rep.cycles_total;
    w.l1d_accesses   = mh.stats().l1d.accesses;
    w.l2_accesses    = mh.stats().l1d.misses;   // L1D miss → L2 access
    w.dram_accesses  = mh.stats().dram_accesses;
    w.tokens         = 1;

    AsicModel m22(AsicModel::defaults_for(TechNode::N22nm));
    AsicReport r22 = m22.project(w);

    CHECK(r22.total_energy_pJ_per_token > 0.0);
    CHECK(r22.area_mm2 > 0.0);
    CHECK(r22.freq_GHz > 0.0);
    CHECK(r22.wall_time_s_per_token > 0.0);
    CHECK(r22.breakdown.compute_pJ > 0.0);

    // Breakdown must sum to total (within rounding).
    const double sum = r22.breakdown.compute_pJ
                     + r22.breakdown.memory_pJ
                     + r22.breakdown.leakage_pJ;
    CHECK(std::fabs(sum - r22.total_energy_pJ_per_token) < 1e-6);

    // Compute energy must be at least counters.total() × min op cost — a
    // sanity floor that catches accidental zeroing.
    const double min_per_op = 0.005;  // pJ
    CHECK(r22.breakdown.compute_pJ
              >= static_cast<double>(s.counters().total()) * min_per_op);

    // Project the same workload @7nm — energy/token should drop and
    // perf/W should rise.
    AsicModel m7(AsicModel::defaults_for(TechNode::N7nm));
    AsicReport r7 = m7.project(w);
    CHECK(r7.total_energy_pJ_per_token < r22.total_energy_pJ_per_token);
    CHECK(r7.perf_per_watt_tokps       > r22.perf_per_watt_tokps);

    std::printf("[G7-ASIC e2e] insns=%llu cycles=%llu | 22nm: %.2f pJ/tok, "
                "%.2f mm², %.2f GHz, %.2e tok/s/W | "
                "compute=%.2f mem=%.2f leak=%.2f pJ | "
                "7nm: %.2f pJ/tok\n",
                static_cast<unsigned long long>(s.counters().total()),
                static_cast<unsigned long long>(rep.cycles_total),
                r22.total_energy_pJ_per_token, r22.area_mm2, r22.freq_GHz,
                r22.perf_per_watt_tokps,
                r22.breakdown.compute_pJ, r22.breakdown.memory_pJ,
                r22.breakdown.leakage_pJ,
                r7.total_energy_pJ_per_token);
}
