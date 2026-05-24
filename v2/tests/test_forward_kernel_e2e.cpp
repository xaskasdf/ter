// G5b: end-to-end microarch trace of a real ternary kernel.
// Runs tk_matmul_b_9t (substrate-native 27-lane ternary dot-product) under
// the full Layer-1 stack: pipeline + RAW hazards + L1D/L2/DRAM hierarchy +
// MSHRs + branch predictor. Reports cycles, IPC, hit rates, MSHR stalls,
// BP accuracy. This is the smallest "real ternary computation" end-to-end
// sanity for the simulator — every TVMAC operates on trits in {-1,0,+1}.
//
// Does NOT load a real model (GGUF). For BitNet 2B-4T / Llama 1B forward,
// see the harness CLI in G8 (heavy runs go to Ryzen+3090 via SSH).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/numfmt.hpp>
#include <tersim/bp_model.hpp>
#include <tersim/cache_model.hpp>
#include <tersim/cycle_model.hpp>
#include <tersim/trace.hpp>

#ifndef TER_KERNELS_DIR
#error "TER_KERNELS_DIR must be defined by the build system"
#endif

#include <cstdio>
#include <vector>

using namespace ter;
using namespace tersim;

namespace {

// Reusable tracer: derives mem_addr from src reg, branch info from pc delta.
auto make_tracer(PipelineModel& pipe) {
    return [&pipe](std::uint64_t pc, const Instr& instr, const Sim& sim) {
        InstrTrace t;
        t.pc   = pc;
        t.op   = instr.op;
        t.dst  = static_cast<std::uint8_t>(instr.dst);
        t.src1 = static_cast<std::uint8_t>(instr.src1);
        t.src2 = static_cast<std::uint8_t>(instr.src2);
        t.imm  = instr.imm;

        // Memory address from register operand. Trit-word index → byte addr
        // assuming 8B per word (Word27 packed in 64-bit int).
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

        // Branch: pc jumped vs advanced by 1?
        const auto next_pc = static_cast<std::uint64_t>(sim.regs().pc().to_int());
        t.branch_taken  = (next_pc != pc + 1);
        t.branch_target = t.branch_taken ? next_pc : pc + 1;

        pipe.on_retire(t);
    };
}

}  // namespace

TEST_CASE("G5b: tk_matmul_b_9t — single tile, full Layer-1 stack") {
    Sim s(1024);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    KernelId id = kt.find("tk_matmul_b_9t");
    REQUIRE(id.valid);

    // 27 trit-valued inputs in {-1, 0, +1} — genuine ternary data.
    constexpr int K = 27;
    std::vector<int> X(K), W(K);
    for (int i = 0; i < K; ++i) {
        X[i] = (i % 3) - 1;          // -1, 0, +1 cycling
        W[i] = ((i * 5) % 3) - 1;
    }
    for (int i = 0; i < K; ++i) {
        s.mem().store_word(static_cast<size_t>(200 + i), Word27::from_int(X[i]));
        s.mem().store_word(static_cast<size_t>(300 + i), Word27::from_int(W[i]));
    }

    int64_t expected = 0;
    for (int i = 0; i < K; ++i) expected += int64_t{X[i]} * int64_t{W[i]};

    // Attach the full microarch stack.
    PipelineConfig pcfg;
    PipelineModel pipe(pcfg);
    MemHierarchyConfig mc; mc.mshrs = 8;
    MemHierarchy mh(mc);
    BpConfig bcfg; bcfg.ghr_history_bits = 0;   // bimodal — small-kernel friendly
    BranchPredictor bp(bcfg);
    pipe.set_mem_hierarchy(&mh);
    pipe.set_branch_predictor(&bp);
    s.set_tracer(make_tracer(pipe));

    std::vector<int64_t> args = {200, 300, 500, 0, 0, 0, 0};
    s.call_kernel(kt, id, args);

    int64_t got = s.mem().load_word(500).to_int();
    CHECK(got == expected);

    // Functional: kernel uses 1 TVMAC + 1 TVSUM. Trace must agree with
    // OpCounters exactly.
    const auto& rep = pipe.report();
    CHECK(s.counters().get(Opcode::TVMAC) == 1);
    CHECK(s.counters().get(Opcode::TVSUM) == 1);
    CHECK(rep.insns_total == static_cast<std::uint64_t>(
              s.counters().total()));

    // Microarch: at least some L1D activity (TVLOADs for X[27] + W[27]).
    CHECK(mh.stats().l1d.accesses > 0);
    // Any branches must be classified by BP.
    CHECK(bp.stats().branches == rep.insns_by_fu[static_cast<size_t>(FuKind::Branch)]);

    std::printf(
        "[G5b 27-lane matmul] cycles=%llu insns=%llu IPC=%.3f | L1D %.1f%% hit "
        "(%llu acc) | L2 %.1f%% hit | DRAM=%llu | MSHR stall=%llu | "
        "BP %.1f%% acc (%llu br)\n",
        static_cast<unsigned long long>(rep.cycles_total),
        static_cast<unsigned long long>(rep.insns_total),
        rep.ipc(),
        100.0 * mh.stats().l1d.hit_rate(),
        static_cast<unsigned long long>(mh.stats().l1d.accesses),
        100.0 * mh.stats().l2.hit_rate(),
        static_cast<unsigned long long>(mh.stats().dram_accesses),
        static_cast<unsigned long long>(rep.mshr_stall_cycles),
        100.0 * bp.stats().accuracy(),
        static_cast<unsigned long long>(bp.stats().branches));
}

TEST_CASE("G5b: tk_matmul_b_9t — 64 tiles (1×1728 @ 1728×1), steady-state metrics") {
    // 64 calls accumulating into a single output. Stresses kernel reuse +
    // working-set behavior (X/W lines hit L1D after warmup; output line
    // stays hot across all 64 iters).
    Sim s(8192);
    KernelTable kt;
    install_default_kernels(s, kt, TER_KERNELS_DIR);
    KernelId id = kt.find("tk_matmul_b_9t");
    REQUIRE(id.valid);

    constexpr int TILES = 64;
    constexpr int K     = 27;
    // Layout: X[0..1727] at [100..1827], W[0..1727] at [2000..3727], ya at 4000.
    // (Earlier 100/1024/2000 overlapped: X[924..] pisaba W[0..].)
    constexpr int xa = 100, wa = 2000, ya = 4000;

    std::vector<int> X(TILES * K), W(TILES * K);
    int64_t expected = 0;
    for (int i = 0; i < TILES * K; ++i) {
        X[i] = ((i * 7) % 3) - 1;
        W[i] = ((i * 11) % 3) - 1;
        expected += int64_t{X[i]} * int64_t{W[i]};
        s.mem().store_word(static_cast<size_t>(xa + i), Word27::from_int(X[i]));
        s.mem().store_word(static_cast<size_t>(wa + i), Word27::from_int(W[i]));
    }

    PipelineConfig pcfg;
    PipelineModel pipe(pcfg);
    MemHierarchyConfig mc; mc.mshrs = 8;
    MemHierarchy mh(mc);
    BpConfig bcfg; bcfg.ghr_history_bits = 0;
    BranchPredictor bp(bcfg);
    pipe.set_mem_hierarchy(&mh);
    pipe.set_branch_predictor(&bp);
    s.set_tracer(make_tracer(pipe));

    int64_t got = 0;
    for (int t = 0; t < TILES; ++t) {
        std::vector<int64_t> args = {xa + t * K, wa + t * K, ya, 0, 0, 0, 0};
        s.call_kernel(kt, id, args);
        got += s.mem().load_word(static_cast<size_t>(ya)).to_int();
    }
    CHECK(got == expected);

    // 64 invocations × 1 TVMAC each.
    CHECK(s.counters().get(Opcode::TVMAC) == TILES);
    CHECK(s.counters().get(Opcode::TVSUM) == TILES);

    const auto& rep = pipe.report();
    // After warmup the kernel code lives in L1D (re-used every call); the X/W
    // tiles are mostly cold (each tile = new line). Expect non-trivial DRAM.
    CHECK(mh.stats().l1d.accesses > 0);
    CHECK(mh.stats().l1d.hit_rate() > 0.0);

    std::printf(
        "[G5b 64-tile matmul] cycles=%llu insns=%llu IPC=%.3f | "
        "L1D hit=%.1f%% (%llu acc) | L2 hit=%.1f%% | DRAM=%llu | "
        "MSHR stall=%llu cyc | BP acc=%.1f%% (%llu br, %llu miss) | "
        "FU mix: scalar=%llu vec=%llu mem=%llu branch=%llu\n",
        static_cast<unsigned long long>(rep.cycles_total),
        static_cast<unsigned long long>(rep.insns_total),
        rep.ipc(),
        100.0 * mh.stats().l1d.hit_rate(),
        static_cast<unsigned long long>(mh.stats().l1d.accesses),
        100.0 * mh.stats().l2.hit_rate(),
        static_cast<unsigned long long>(mh.stats().dram_accesses),
        static_cast<unsigned long long>(rep.mshr_stall_cycles),
        100.0 * bp.stats().accuracy(),
        static_cast<unsigned long long>(bp.stats().branches),
        static_cast<unsigned long long>(bp.stats().mispredicts),
        static_cast<unsigned long long>(rep.insns_by_fu[
            static_cast<size_t>(FuKind::Scalar)]),
        static_cast<unsigned long long>(rep.insns_by_fu[
            static_cast<size_t>(FuKind::Vector)]),
        static_cast<unsigned long long>(rep.insns_by_fu[
            static_cast<size_t>(FuKind::Memory)]),
        static_cast<unsigned long long>(rep.insns_by_fu[
            static_cast<size_t>(FuKind::Branch)]));
}

TEST_CASE("G5b: cache attached vs unattached — DRAM tax on 64-tile matmul") {
    // Same kernel, twice: once with no cache (default mem_extra_latency=2 ≈
    // perfect L1D), once with the full hierarchy (cold-load misses hit DRAM).
    // Cycle delta is the substrate's data-motion cost — the headline number
    // for the bandwidth claim in §5.2 of the paper.
    auto run_once = [](bool with_cache) -> std::pair<std::uint64_t, std::uint64_t> {
        Sim s(8192);
        KernelTable kt;
        install_default_kernels(s, kt, TER_KERNELS_DIR);
        KernelId id = kt.find("tk_matmul_b_9t");
        constexpr int TILES = 64, K = 27;
        constexpr int xa = 100, wa = 2000, ya = 4000;
        for (int i = 0; i < TILES * K; ++i) {
            s.mem().store_word(static_cast<size_t>(xa + i),
                               Word27::from_int(((i * 7) % 3) - 1));
            s.mem().store_word(static_cast<size_t>(wa + i),
                               Word27::from_int(((i * 11) % 3) - 1));
        }
        PipelineConfig pcfg;
        PipelineModel pipe(pcfg);
        MemHierarchyConfig mc; mc.mshrs = 8;
        MemHierarchy mh(mc);
        if (with_cache) pipe.set_mem_hierarchy(&mh);
        s.set_tracer(make_tracer(pipe));
        for (int t = 0; t < TILES; ++t) {
            std::vector<int64_t> args = {xa + t * K, wa + t * K, ya, 0, 0, 0, 0};
            s.call_kernel(kt, id, args);
        }
        return {pipe.report().cycles_total, mh.stats().dram_accesses};
    };

    auto [cyc_nocache, dram_nocache] = run_once(false);
    auto [cyc_cache,   dram_cache]   = run_once(true);

    CHECK(dram_nocache == 0);    // no hierarchy attached → no DRAM tracked
    CHECK(dram_cache   >  0);    // real cold misses recorded
    CHECK(cyc_cache    >  cyc_nocache);   // cache attached adds miss tax

    std::printf("[G5b cache tax] no-hier=%llu cyc | with-hier=%llu cyc "
                "(+%lld = %.1f%% slowdown) | DRAM=%llu accesses\n",
                static_cast<unsigned long long>(cyc_nocache),
                static_cast<unsigned long long>(cyc_cache),
                static_cast<long long>(cyc_cache) - static_cast<long long>(cyc_nocache),
                100.0 * (static_cast<double>(cyc_cache) /
                         static_cast<double>(cyc_nocache) - 1.0),
                static_cast<unsigned long long>(dram_cache));
}
