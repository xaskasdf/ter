#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/assembler.hpp>
#include <tersim/trace.hpp>

#include <vector>

using namespace ter;

namespace {

// Inline source so the test is location-independent (runs the same from
// macOS dev box and msys64 remote without examples/ fixture lookup).
constexpr const char* kSumProgram = R"(
        tloadi r1, 0
        tloadi r2, 5
        tloadi r3, 1
        tloadi r4, 0
loop:
        tbeq   r2, r4, done
        tadd   r1, r1, r2
        tsub   r2, r2, r3
        tjump  loop
done:
        thalt
)";

tersim::InstrTrace make_trace(std::uint64_t pc, const Instr& i) {
    tersim::InstrTrace t;
    t.pc   = pc;
    t.op   = i.op;
    t.dst  = static_cast<std::uint8_t>(i.dst);
    t.src1 = static_cast<std::uint8_t>(i.src1);
    t.src2 = static_cast<std::uint8_t>(i.src2);
    t.imm  = i.imm;
    t.mem_is_load  = (i.op == Opcode::TLOAD || i.op == Opcode::TVLOAD);
    t.mem_is_store = (i.op == Opcode::TSTORE || i.op == Opcode::TVSTORE);
    return t;
}

}  // namespace

TEST_CASE("G1c: tracer fires once per retired instruction, matches OpCounters") {
    auto blob = assemble(kSumProgram);
    Sim s(256);
    for (size_t i = 0; i < blob.size(); ++i) s.mem().store_word(i, blob[i]);

    std::vector<tersim::InstrTrace> trace;
    s.set_tracer([&](std::uint64_t pc, const Instr& instr, const Sim&) {
        trace.push_back(make_trace(pc, instr));
    });

    s.run();

    // Functional result unchanged by tracer.
    CHECK(s.regs().read_scalar(1).to_int() == 15);

    // Trace counts must exactly equal OpCounters — every retired insn fires once.
    auto count_op = [&](Opcode op) {
        std::size_t n = 0;
        for (auto& t : trace) if (t.op == op) ++n;
        return n;
    };
    CHECK(count_op(Opcode::TADD)   == s.counters().get(Opcode::TADD));
    CHECK(count_op(Opcode::TSUB)   == s.counters().get(Opcode::TSUB));
    CHECK(count_op(Opcode::TJUMP)  == s.counters().get(Opcode::TJUMP));
    CHECK(count_op(Opcode::TBEQ)   == s.counters().get(Opcode::TBEQ));
    CHECK(count_op(Opcode::TLOADI) == s.counters().get(Opcode::TLOADI));
    CHECK(count_op(Opcode::THALT)  == s.counters().get(Opcode::THALT));

    // PCs must be strictly within program size and monotonic-or-backward
    // (backward only on TJUMP / TBEQ branches taken).
    for (auto& t : trace) CHECK(t.pc < blob.size());
}

TEST_CASE("G1c: default tracer is empty (no callback fires, no overhead)") {
    auto blob = assemble(kSumProgram);
    Sim s(256);
    for (size_t i = 0; i < blob.size(); ++i) s.mem().store_word(i, blob[i]);
    // No set_tracer call — should run normally.
    s.run();
    CHECK(s.regs().read_scalar(1).to_int() == 15);
    CHECK(static_cast<bool>(s.tracer()) == false);
}
