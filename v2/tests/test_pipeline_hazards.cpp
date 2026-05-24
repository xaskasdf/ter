#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/assembler.hpp>
#include <tersim/cycle_model.hpp>
#include <tersim/trace.hpp>

using namespace ter;
using namespace tersim;

namespace {

InstrTrace make(Opcode op, int dst = 0, int src1 = 0, int src2 = 0) {
    InstrTrace t;
    t.op = op;
    t.dst  = static_cast<std::uint8_t>(dst);
    t.src1 = static_cast<std::uint8_t>(src1);
    t.src2 = static_cast<std::uint8_t>(src2);
    return t;
}

}  // namespace

TEST_CASE("G3: RAW chain back-to-back under EX-bypass forwarding") {
    // 5 dependent TADDs: r1=r0+r0; r2=r1+r1; r3=r2+r2; r4=r3+r3; r5=r4+r4.
    // With forwarding: producer result available end of EX1 (issue+4), consumer
    // reads at EX1 (issue+4), gap = 0 → back-to-back issue. cycles = 5+6 = 11.
    PipelineConfig cfg; // defaults: track_raw=true, forwarding=true
    PipelineModel pipe(cfg);

    auto t1 = make(Opcode::TADD, 1, 0, 0); pipe.on_retire(t1);
    auto t2 = make(Opcode::TADD, 2, 1, 1); pipe.on_retire(t2);
    auto t3 = make(Opcode::TADD, 3, 2, 2); pipe.on_retire(t3);
    auto t4 = make(Opcode::TADD, 4, 3, 3); pipe.on_retire(t4);
    auto t5 = make(Opcode::TADD, 5, 4, 4); pipe.on_retire(t5);

    CHECK(t1.cycle_issued == 0);
    CHECK(t2.cycle_issued == 1);   // back-to-back, no stall
    CHECK(t3.cycle_issued == 2);
    CHECK(t4.cycle_issued == 3);
    CHECK(t5.cycle_issued == 4);
    CHECK(pipe.report().cycles_total == 11);
}

TEST_CASE("G3: RAW chain stalls 4c/insn without forwarding") {
    // Same chain, forwarding off. Producer writes RF at WB (issue+6),
    // consumer reads at RR (issue+2). consumer_issue >= producer_issue+4.
    PipelineConfig cfg; cfg.forwarding_enabled = false;
    PipelineModel pipe(cfg);

    auto t1 = make(Opcode::TADD, 1, 0, 0); pipe.on_retire(t1);
    auto t2 = make(Opcode::TADD, 2, 1, 1); pipe.on_retire(t2);
    auto t3 = make(Opcode::TADD, 3, 2, 2); pipe.on_retire(t3);

    CHECK(t1.cycle_issued == 0);
    CHECK(t2.cycle_issued == 4);   // 4-cycle stall
    CHECK(t3.cycle_issued == 8);
    CHECK(pipe.report().cycles_total == 8 + 6 + 1);  // 15
}

TEST_CASE("G3: TVMAC accumulator chain — RMW dep on a0, 1c gap with forwarding") {
    // 5 TVMACs all writing a0; each reads a0 (RMW). With forwarding,
    // producer writes a0 at issue+4+1=5; consumer reads at issue+4;
    // → consumer >= producer+1. Back-to-back issue gap = 1.
    PipelineConfig cfg;
    PipelineModel pipe(cfg);

    InstrTrace ts[5];
    for (int k = 0; k < 5; ++k) {
        ts[k] = make(Opcode::TVMAC, 0, 0, 1);
        pipe.on_retire(ts[k]);
    }
    CHECK(ts[0].cycle_issued == 0);
    CHECK(ts[1].cycle_issued == 1);   // forwarding lets RMW back-to-back
    CHECK(ts[2].cycle_issued == 2);
    CHECK(ts[4].cycle_issued == 4);
    // Last retires at 4 + 6 + 1 = 11; cycles = 12.
    CHECK(pipe.report().cycles_total == 12);
}

TEST_CASE("G3: TVLOAD→TVMAC has 2c gap (mem extra latency forwarded)") {
    // TVLOAD writes v0 at issue + 4 + mem_extra(2) = issue+6.
    // TVMAC reads v0 at consumer_issue+4 → consumer >= producer+2.
    PipelineConfig cfg;
    PipelineModel pipe(cfg);

    auto ld = make(Opcode::TVLOAD, /*dst=v0*/0, /*src1=r0 addr*/0);
    auto mc = make(Opcode::TVMAC,  /*dst=a0*/0, /*src1=v0*/0, /*src2=v1*/1);
    pipe.on_retire(ld);
    pipe.on_retire(mc);

    CHECK(ld.cycle_issued == 0);
    CHECK(mc.cycle_issued == 2);   // 1c stall (next_issue=1, dep=2)
    // mc retires at 2 + 6 + 1 = 9; cycles = 10.
    CHECK(pipe.report().cycles_total == 10);
}

TEST_CASE("G3: TLOAD without forwarding stalls 6c") {
    // TLOAD result at WB = issue + 6 + 2 = issue+8. Consumer reads at RR = issue+2.
    // → consumer_issue >= producer_issue + 6.
    PipelineConfig cfg; cfg.forwarding_enabled = false;
    PipelineModel pipe(cfg);

    auto ld = make(Opcode::TLOAD, 1, 0);
    auto ad = make(Opcode::TADD,  2, 1, 1);
    pipe.on_retire(ld);
    pipe.on_retire(ad);

    CHECK(ld.cycle_issued == 0);
    CHECK(ad.cycle_issued == 6);
}

TEST_CASE("G3: independent stream — no hazards, IPC ~1") {
    // 1000 TADDs but each writes a DIFFERENT register, no chain.
    // Even with track_raw=true and no chain, IPC saturates.
    PipelineConfig cfg;
    PipelineModel pipe(cfg);
    for (int k = 0; k < 1000; ++k) {
        // dst rotates over R0..R25 (avoid 26 = sp); reads R26 (sp-like, no writes)
        auto t = make(Opcode::TADD, k % 26, 26, 26);
        pipe.on_retire(t);
    }
    const auto& rep = pipe.report();
    CHECK(rep.insns_total == 1000);
    CHECK(rep.cycles_total == 1006);  // perfect pipeline
    CHECK(rep.ipc() > 0.99);
}

TEST_CASE("G3: real TVMAC loop with hazards matches hand-derived cycle count") {
    // Same 8-iter loop from G2. Per-iter: TVLOAD@T, TVLOAD@T+1, TVMAC@T+3
    // (1c stall from v1), TSUB@T+4, TBNE@T+5. Next iter @T+6.
    // 4 setup + 8 iters × 6 cycles + THALT issue + drain.
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

    PipelineConfig cfg;   // defaults: hazards on, forwarding on
    PipelineModel pipe(cfg);
    s.set_tracer([&](std::uint64_t, const Instr& instr, const Sim&) {
        InstrTrace t = make(instr.op, instr.dst, instr.src1, instr.src2);
        pipe.on_retire(t);
    });
    s.run();

    const auto& rep = pipe.report();
    CHECK(rep.insns_total == 45);
    // Setups 0..3, 8 iters × 6c starting at 4, THALT at 52, drain 6 → 59.
    CHECK(rep.cycles_total == 59);
    // IPC ~0.76 (vs 0.88 in G2 no-hazards baseline) — captures the real loop cost.
    CHECK(rep.ipc() > 0.75);
    CHECK(rep.ipc() < 0.78);
}
