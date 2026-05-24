#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/assembler.hpp>
#include <tersim/cycle_model.hpp>
#include <tersim/trace.hpp>

using namespace ter;
using namespace tersim;

namespace {

// Tight 8-iteration TVMAC loop: load v0,v1, mac into a0, branch.
// Body per-iter: 1 TVLOAD + 1 TVLOAD + 1 TVMAC + 1 TSUB (counter) + 1 TBNE.
// 5 ops × 8 iters = 40 body insns + 4 setup insns + final THALT = 45 retired.
constexpr const char* kTvmacLoop = R"(
        tloadi r1, 64       ; base addr A
        tloadi r2, 128      ; base addr B
        tloadi r3, 8        ; loop count
        tloadi r4, 1
loop:
        tvload v0, r1
        tvload v1, r2
        tvmac  a0, v0, v1
        tsub   r3, r3, r4
        tbne   r3, r0, loop
        thalt
)";

InstrTrace make_trace(std::uint64_t pc, const Instr& i) {
    InstrTrace t;
    t.pc   = pc;
    t.op   = i.op;
    t.dst  = static_cast<std::uint8_t>(i.dst);
    t.src1 = static_cast<std::uint8_t>(i.src1);
    t.src2 = static_cast<std::uint8_t>(i.src2);
    t.imm  = i.imm;
    return t;
}

}  // namespace

TEST_CASE("G2: pipeline retires every instr, IPC -> 1 as N grows (no hazards)") {
    auto blob = assemble(kTvmacLoop);
    Sim s(512);
    for (size_t i = 0; i < blob.size(); ++i) s.mem().store_word(i, blob[i]);

    // G2 baseline: hazard tracking disabled (G3 adds it back).
    PipelineConfig cfg; cfg.track_raw = false;
    PipelineModel pipe(cfg);
    s.set_tracer([&](std::uint64_t pc, const Instr& instr, const Sim&) {
        auto t = make_trace(pc, instr);
        pipe.on_retire(t);
    });

    s.run();

    const auto& rep = pipe.report();

    // Functional sanity — loop ran 8 iters, counter at 0.
    CHECK(s.regs().read_scalar(3).to_int() == 0);

    // 4 setup + 8×5 body + 1 halt = 45 retired insns.
    CHECK(rep.insns_total == 45);

    // FU mix: 16 vector (8 TVLOAD + 8 TVLOAD = 16 vector loads; wait, TVLOAD
    // is Memory not Vector). Recompute: scalar 4 setup TLOADI + 8 TSUB = 12;
    // vector 8 TVMAC = 8; memory 16 TVLOAD; branch 8 TBNE; control 1 THALT.
    CHECK(rep.insns_by_fu[static_cast<size_t>(FuKind::Scalar)]  == 12);
    CHECK(rep.insns_by_fu[static_cast<size_t>(FuKind::Vector)]  == 8);
    CHECK(rep.insns_by_fu[static_cast<size_t>(FuKind::Memory)]  == 16);
    CHECK(rep.insns_by_fu[static_cast<size_t>(FuKind::Branch)]  == 8);
    CHECK(rep.insns_by_fu[static_cast<size_t>(FuKind::Control)] == 1);

    // G2 timing: single-issue, no hazards, fully pipelined.
    // cycles ≈ N + (depth-1) + tail_extra_latency.
    // The last instr is THALT (control, extra=0). The TVMAC adds extra=1 but
    // it's mid-loop; pipelining absorbs it back-to-back. So tail = depth-1 = 6.
    // Expected: 45 + 6 = 51 cycles.
    CHECK(rep.cycles_total == 51);

    // IPC ≈ 45/51 ≈ 0.882; will saturate to 1 for longer streams.
    CHECK(rep.ipc() > 0.85);
    CHECK(rep.ipc() < 1.0);
}

TEST_CASE("G2: longer stream pushes IPC closer to 1") {
    // Synthesize a 1000-insn stream of TADD; expected cycles = 1000+6 = 1006;
    // IPC = 1000/1006 ≈ 0.994.
    PipelineModel pipe;
    for (int k = 0; k < 1000; ++k) {
        InstrTrace t;
        t.pc = static_cast<std::uint64_t>(k);
        t.op = Opcode::TADD;
        pipe.on_retire(t);
    }
    const auto& rep = pipe.report();
    CHECK(rep.insns_total == 1000);
    CHECK(rep.cycles_total == 1006);
    CHECK(rep.ipc() > 0.99);
}

TEST_CASE("G2: TVMAC extra latency lengthens drain when last") {
    PipelineModel pipe;
    InstrTrace t1; t1.op = Opcode::TADD;  pipe.on_retire(t1);
    InstrTrace t2; t2.op = Opcode::TVMAC; pipe.on_retire(t2);
    // TADD: issued=0, retired=6. TVMAC: issued=1, retired=1+6+1=8. Total=9 cycles.
    CHECK(t1.cycle_issued == 0);
    CHECK(t1.cycle_retired == 6);
    CHECK(t2.cycle_issued == 1);
    CHECK(t2.cycle_retired == 8);
    CHECK(pipe.report().cycles_total == 9);
}
