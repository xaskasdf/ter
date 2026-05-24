#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <tersim/cache_model.hpp>
#include <tersim/cycle_model.hpp>
#include <tersim/trace.hpp>

using namespace tersim;
using namespace ter;

namespace {

InstrTrace mem(Opcode op, std::uint64_t addr, int dst = 0, int src1 = 0, int src2 = 0) {
    InstrTrace t;
    t.op       = op;
    t.dst      = static_cast<std::uint8_t>(dst);
    t.src1     = static_cast<std::uint8_t>(src1);
    t.src2     = static_cast<std::uint8_t>(src2);
    t.mem_addr = addr;
    t.mem_is_load  = (op == Opcode::TLOAD  || op == Opcode::TVLOAD);
    t.mem_is_store = (op == Opcode::TSTORE || op == Opcode::TVSTORE);
    return t;
}

}  // namespace

TEST_CASE("G4: L1D — first access to line misses, repeated accesses hit") {
    CacheConfig cfg;   // 32K, 64B lines, 8-way, hit lat 3 (defaults)
    Cache l1d(cfg);

    CHECK(l1d.access(0x1000) == false);   // cold miss
    CHECK(l1d.access(0x1000) == true);    // hot
    CHECK(l1d.access(0x1008) == true);    // same 64B line
    CHECK(l1d.access(0x103F) == true);    // last byte of same line
    CHECK(l1d.access(0x1040) == false);   // next line, miss
    CHECK(l1d.stats().accesses == 5);
    CHECK(l1d.stats().hits   == 3);
    CHECK(l1d.stats().misses == 2);
}

TEST_CASE("G4: L1D — stride-1 scan over 32 KiB has 1/8 miss rate (one per line)") {
    Cache l1d(CacheConfig{});
    // 32 KiB scan, 8-byte stride: 4096 accesses, 64 per line, 1 miss/line = 512 misses.
    constexpr std::uint64_t bytes = 32 * 1024;
    for (std::uint64_t a = 0; a < bytes; a += 8) (void)l1d.access(a);
    const auto& s = l1d.stats();
    CHECK(s.accesses == 4096);
    CHECK(s.misses   == 512);
    CHECK(s.hits     == 4096 - 512);
}

TEST_CASE("G4: working set > L1D — sweeps thrash, drop to L2") {
    MemHierarchy mh;   // defaults: L1D 32K, L2 512K, DRAM 80c
    // Stride-line scan over 256 KiB (fits in L2 but not L1D); two sweeps.
    constexpr std::uint64_t bytes = 256 * 1024;
    for (int pass = 0; pass < 2; ++pass) {
        for (std::uint64_t a = 0; a < bytes; a += 64) (void)mh.access(a, true);
    }
    const auto& s = mh.stats();
    // Total accesses = 2 * (256K/64) = 8192. First pass: all miss L1D, L2 fills.
    // Second pass: L1D still misses (256K > 32K), L2 hits.
    CHECK(s.l1d.accesses == 8192);
    CHECK(s.l1d.misses   == 8192);
    CHECK(s.l2.accesses  == 8192);
    // First pass: all 4096 lines miss L2 (cold) and go to DRAM.
    // Second pass: L2 holds 512K worth = 8192 lines, but only 4096 unique → all hit.
    CHECK(s.dram_accesses == 4096);
    CHECK(s.l2.hits      == 4096);
}

TEST_CASE("G4: pipeline integrates cache — TVLOAD latency depends on hit/miss") {
    MemHierarchy mh;
    PipelineConfig cfg;
    PipelineModel pipe(cfg);
    pipe.set_mem_hierarchy(&mh);

    // 1st TVLOAD: cold miss → DRAM = 80c total mem latency → extra = 79.
    // Writes v0 at issue + 4 + 79 = issue + 83.
    auto t1 = mem(Opcode::TVLOAD, 0x1000, /*dst=v0*/0, /*src1=r0 addr*/0);
    pipe.on_retire(t1);
    CHECK(t1.cycle_issued  == 0);
    CHECK(t1.cycle_retired == 0u + 6 + 79);   // depth-1 + extra = 85
    CHECK(pipe.report().cycles_total == 86);

    // 2nd TVLOAD same line: L1D hit → 3c → extra = 2.
    auto t2 = mem(Opcode::TVLOAD, 0x1000, /*dst=v1*/1, /*src1=r0*/0);
    pipe.on_retire(t2);
    // Issue can start at next_issue=1 (no RAW: v1 != src). Retires at 1+6+2 = 9.
    // But last_retire from t1 was 85, so cycles_total = max(85+1, 9+1) = 86.
    CHECK(t2.cycle_issued == 1);
    CHECK(t2.cycle_retired == 9);
    CHECK(pipe.report().cycles_total == 86);
}

TEST_CASE("G4: compute-bound vs memory-bound — divergent IPC curves") {
    // Compute-bound: a tight TADD chain on registers (no memory). IPC ~1.
    {
        PipelineConfig cfg;
        PipelineModel pipe(cfg);
        for (int k = 0; k < 200; ++k) {
            InstrTrace t; t.op = Opcode::TADD;
            t.dst  = static_cast<std::uint8_t>(k % 25);   // rotate, avoid r26
            t.src1 = 26; t.src2 = 26;                     // read-only base
            pipe.on_retire(t);
        }
        CHECK(pipe.report().ipc() > 0.97);
    }

    // Memory-bound: TVLOADs sweeping a working set that misses L1D every line
    // (stride = 64B, 2 MiB region → far larger than L2 too, so each line is DRAM).
    {
        MemHierarchy mh;
        PipelineConfig cfg;
        PipelineModel pipe(cfg);
        pipe.set_mem_hierarchy(&mh);
        for (int k = 0; k < 200; ++k) {
            auto t = mem(Opcode::TVLOAD, /*addr*/0x4000 + 64ull * static_cast<unsigned>(k),
                         /*dst*/k % 9, /*src1*/0);
            pipe.on_retire(t);
        }
        // Every access is a cold miss → DRAM (80c). With single-issue and no
        // dependent reader, loads can pipeline back-to-back (no MSHR model yet),
        // so IPC is dominated by the in-pipe latency at the tail (drain ~84c).
        // For N=200: cycles ≈ 200 + 84 = 284. IPC ≈ 0.70.
        const auto& rep = pipe.report();
        CHECK(rep.cycles_total > 280);
        CHECK(rep.cycles_total < 290);
        CHECK(rep.ipc() < 0.75);
        CHECK(mh.stats().dram_accesses == 200);
    }
}
