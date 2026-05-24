#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/assembler.hpp>
#include <tersim/bp_model.hpp>
#include <tersim/cache_model.hpp>
#include <tersim/cycle_model.hpp>
#include <tersim/trace.hpp>

using namespace ter;
using namespace tersim;

namespace {

InstrTrace mem(Opcode op, std::uint64_t addr, int dst = 0, int src1 = 0) {
    InstrTrace t;
    t.op = op; t.dst = static_cast<std::uint8_t>(dst);
    t.src1 = static_cast<std::uint8_t>(src1);
    t.mem_addr = addr;
    t.mem_is_load = (op == Opcode::TLOAD || op == Opcode::TVLOAD);
    return t;
}

}  // namespace

// ------- MSHR -------

TEST_CASE("G5: MSHR unlimited (mshrs=0) — 16 DRAM misses pipeline back-to-back") {
    MemHierarchyConfig mc; mc.mshrs = 0;
    MemHierarchy mh(mc);
    PipelineConfig pc;
    PipelineModel pipe(pc);
    pipe.set_mem_hierarchy(&mh);

    for (int k = 0; k < 16; ++k) {
        auto t = mem(Opcode::TVLOAD, /*unique line*/ 0x10000 + 64ull * static_cast<unsigned>(k),
                     k % 9);
        pipe.on_retire(t);
    }
    CHECK(mh.stats().dram_accesses == 16);
    CHECK(mh.stats().mshr_stall_cycles == 0);
    CHECK(pipe.report().mshr_stall_cycles == 0);
}

TEST_CASE("G5: MSHR=8 caps in-flight misses, 9th+ stalls until oldest frees") {
    MemHierarchyConfig mc; mc.mshrs = 8;
    MemHierarchy mh(mc);
    PipelineConfig pc;
    PipelineModel pipe(pc);
    pipe.set_mem_hierarchy(&mh);

    // 16 cold DRAM misses, distinct lines (no L2 reuse). Each load takes 80c.
    // Misses 0..7 issue at cycles 0..7, free at 80..87.
    // Miss 8 issues at next_issue=8, all 8 MSHRs busy (free at 80..87, all >8).
    //   Oldest free = 80 → issue pushed to 80, free at 80+80=160.
    // Miss 9 issues at next_issue=81; mshrs busy with free=[81,82,...,87,160].
    //   Oldest=81 → already ≤ proposed, prune → 7 in-flight, OK. Free at 81+80=161.
    // ... and so on.
    for (int k = 0; k < 16; ++k) {
        auto t = mem(Opcode::TVLOAD, 0x20000 + 64ull * static_cast<unsigned>(k), k % 9);
        pipe.on_retire(t);
    }
    CHECK(mh.stats().dram_accesses == 16);
    // Miss 8 should accumulate stall_cycles = 80 - 8 = 72. Subsequent misses
    // shouldn't add (they prune naturally). So total stall = 72.
    CHECK(mh.stats().mshr_stall_cycles == 72);
    CHECK(pipe.report().mshr_stall_cycles == 72);
}

// ------- BP -------

TEST_CASE("G5: BP perfect — zero mispredicts, zero penalty") {
    BpConfig bcfg; bcfg.perfect = true;
    BranchPredictor bp(bcfg);
    PipelineConfig pc;
    PipelineModel pipe(pc);
    pipe.set_branch_predictor(&bp);

    // 10 backward branches, all taken.
    for (int k = 0; k < 10; ++k) {
        InstrTrace t;
        t.op = Opcode::TBNE;
        t.pc = 100;
        t.branch_taken = true;
        t.branch_target = 50;
        pipe.on_retire(t);
    }
    CHECK(bp.stats().branches    == 10);
    CHECK(bp.stats().correct     == 10);
    CHECK(bp.stats().mispredicts == 0);
    CHECK(pipe.report().bp_penalty_cycles == 0);
}

TEST_CASE("G5: bimodal learns a single hot branch in one update") {
    BpConfig bcfg; bcfg.ghr_history_bits = 0;   // bimodal — same PHT slot reused
    BranchPredictor bp(bcfg);

    // Branch at pc=100, target=50, taken=true. With bimodal, the same PHT
    // entry is updated every iter; counter saturates after 1 mispredict.
    bool first  = bp.predict_and_update(100, /*taken*/true, /*target*/50);
    bool second = bp.predict_and_update(100, true, 50);
    bool third  = bp.predict_and_update(100, true, 50);
    bool fourth = bp.predict_and_update(100, true, 50);
    CHECK(first  == false);   // weak not-taken initially → mispredict, counter 1→2
    CHECK(second == true);    // counter=2 → predict taken, correct, BTB[100] now valid
    CHECK(third  == true);
    CHECK(fourth == true);
    CHECK(bp.stats().correct == 3);
    CHECK(bp.stats().mispredicts == 1);
}

TEST_CASE("G5: end-to-end Sim — TVMAC loop with BP recovers after warmup") {
    constexpr const char* kLoop = R"(
        tloadi r1, 64
        tloadi r2, 128
        tloadi r3, 8
        tloadi r4, 1
loop:
        tvload v0, r1
        tvload v1, r2
        tvmac  a0, v0, v1
        tsub   r3, r3, r4
        tbne   r3, r0, loop
        thalt
)";
    auto blob = assemble(kLoop);
    Sim s(512);
    for (size_t i = 0; i < blob.size(); ++i) s.mem().store_word(i, blob[i]);

    PipelineConfig pcfg;
    PipelineModel pipe(pcfg);
    BpConfig bcfg;
    BranchPredictor bp(bcfg);
    pipe.set_branch_predictor(&bp);

    s.set_tracer([&](std::uint64_t pc, const Instr& instr, const Sim& sim) {
        InstrTrace t;
        t.pc = pc;
        t.op = instr.op;
        t.dst  = static_cast<std::uint8_t>(instr.dst);
        t.src1 = static_cast<std::uint8_t>(instr.src1);
        t.src2 = static_cast<std::uint8_t>(instr.src2);
        // Branch info: if pc advanced by 1 → fall-through; else → taken.
        const auto next_pc = static_cast<std::uint64_t>(sim.regs().pc().to_int());
        t.branch_taken  = (next_pc != pc + 1);
        t.branch_target = t.branch_taken ? next_pc : pc + 1;
        pipe.on_retire(t);
    });
    s.run();

    // 8 backedges (TBNE taken first 7 iters, not-taken on 8th).
    CHECK(bp.stats().branches == 8);
    // gshare with 12-bit GHR: each iter shifts GHR, so each branch hits a
    // DIFFERENT PHT slot (all initialized to weak not-taken). All 7 taken
    // branches mispredict; the 8th (not-taken) hits another fresh slot whose
    // counter=1 predicts not-taken → correct.
    CHECK(bp.stats().mispredicts == 7);
    CHECK(bp.stats().correct     == 1);
    // Penalty: 7 × 4 = 28.
    CHECK(pipe.report().bp_penalty_cycles == 28);
}

TEST_CASE("G5: bimodal (ghr_history_bits=0) DOES stabilize a tight loop") {
    constexpr const char* kLoop = R"(
        tloadi r1, 64
        tloadi r2, 128
        tloadi r3, 8
        tloadi r4, 1
loop:
        tvload v0, r1
        tvload v1, r2
        tvmac  a0, v0, v1
        tsub   r3, r3, r4
        tbne   r3, r0, loop
        thalt
)";
    auto blob = assemble(kLoop);
    Sim s(512);
    for (size_t i = 0; i < blob.size(); ++i) s.mem().store_word(i, blob[i]);

    PipelineConfig pcfg;
    PipelineModel pipe(pcfg);
    BpConfig bcfg; bcfg.ghr_history_bits = 0;   // bimodal — same PHT slot always
    BranchPredictor bp(bcfg);
    pipe.set_branch_predictor(&bp);

    s.set_tracer([&](std::uint64_t pc, const Instr& instr, const Sim& sim) {
        InstrTrace t;
        t.pc = pc; t.op = instr.op;
        t.dst  = static_cast<std::uint8_t>(instr.dst);
        t.src1 = static_cast<std::uint8_t>(instr.src1);
        t.src2 = static_cast<std::uint8_t>(instr.src2);
        const auto next_pc = static_cast<std::uint64_t>(sim.regs().pc().to_int());
        t.branch_taken  = (next_pc != pc + 1);
        t.branch_target = t.branch_taken ? next_pc : pc + 1;
        pipe.on_retire(t);
    });
    s.run();

    // Iter 1: counter=1 (weak not-taken), predict N, actual T → mispredict, counter→2.
    // Iter 2: counter=2, predict T, actual T → correct, counter→3.
    // Iter 3..7: counter=3 (saturated), predict T → correct (5 correct).
    // Iter 8: counter=3, predict T, actual N → mispredict, counter→2.
    CHECK(bp.stats().branches == 8);
    CHECK(bp.stats().mispredicts == 2);
    CHECK(bp.stats().correct     == 6);
    CHECK(pipe.report().bp_penalty_cycles == 8);
}

TEST_CASE("G5: combined — TVMAC loop with BP + L1D-cached working set (warm)") {
    // Set up TVMAC loop, attach BP + MemHierarchy with MSHR=8.
    // Working set fits in L1D, so all TVLOADs hit (3c each); BP learns loop.
    // This is the realistic "small kernel, hot cache" steady-state.
    constexpr const char* kLoop = R"(
        tloadi r1, 64
        tloadi r2, 128
        tloadi r3, 4
        tloadi r4, 1
loop:
        tvload v0, r1
        tvload v1, r2
        tvmac  a0, v0, v1
        tsub   r3, r3, r4
        tbne   r3, r0, loop
        thalt
)";
    auto blob = assemble(kLoop);
    Sim s(512);
    for (size_t i = 0; i < blob.size(); ++i) s.mem().store_word(i, blob[i]);

    MemHierarchyConfig mc; mc.mshrs = 8;
    MemHierarchy mh(mc);
    BpConfig bcfg;
    BranchPredictor bp(bcfg);

    PipelineConfig pcfg;
    PipelineModel pipe(pcfg);
    pipe.set_mem_hierarchy(&mh);
    pipe.set_branch_predictor(&bp);

    s.set_tracer([&](std::uint64_t pc, const Instr& instr, const Sim& sim) {
        InstrTrace t;
        t.pc = pc; t.op = instr.op;
        t.dst  = static_cast<std::uint8_t>(instr.dst);
        t.src1 = static_cast<std::uint8_t>(instr.src1);
        t.src2 = static_cast<std::uint8_t>(instr.src2);
        // Compute mem_addr from src1 register for memory ops (TVLOAD reads r[src1]).
        if (instr.op == Opcode::TVLOAD || instr.op == Opcode::TLOAD) {
            t.mem_addr = static_cast<std::uint64_t>(sim.regs().read_scalar(instr.src1).to_int()) * 8;
            t.mem_is_load = true;
        }
        const auto next_pc = static_cast<std::uint64_t>(sim.regs().pc().to_int());
        t.branch_taken  = (next_pc != pc + 1);
        t.branch_target = t.branch_taken ? next_pc : pc + 1;
        pipe.on_retire(t);
    });
    s.run();

    const auto& rep = pipe.report();
    // 4 iters × 5 + 4 setups + 1 halt = 25 insns retired.
    CHECK(rep.insns_total == 25);
    // Two unique 64B lines accessed (r1=64 → addr 512, r2=128 → addr 1024).
    // Each line: 1 cold miss + (4-1) hits over the 4 iters per address... but
    // both addresses fit easily, no thrashing.
    CHECK(mh.stats().dram_accesses == 2);   // 2 cold cap-line misses
    CHECK(mh.stats().l1d.misses    == 2);
    CHECK(mh.stats().l1d.hits      == 6);   // 4 iters × 2 loads − 2 cold = 6
    CHECK(mh.stats().mshr_stall_cycles == 0);  // 2 misses, 8 MSHRs, never full
    // BP: 4 branches, gshare 12-bit GHR — each iter hits different PHT slot.
    // Iters 1-3 (taken) all mispredict; iter 4 (not-taken) hits fresh slot,
    // predicts not-taken → correct.
    CHECK(bp.stats().branches == 4);
    CHECK(bp.stats().mispredicts == 3);
    CHECK(bp.stats().correct     == 1);
}
