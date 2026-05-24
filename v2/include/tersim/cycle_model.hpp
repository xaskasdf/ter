#pragma once
#include <tersim/trace.hpp>
#include <ter/isa.hpp>
#include <array>
#include <cstdint>

namespace tersim {

class MemHierarchy;     // fwd; #include <tersim/cache_model.hpp> from pipeline.cpp
class BranchPredictor;  // fwd; #include <tersim/bp_model.hpp> from pipeline.cpp

enum class FuKind : std::uint8_t {
    Scalar  = 0,  // TADD, TSUB, TNEG, TAND3, ..., TLOADI, TCMP, TSIGN
    Vector  = 1,  // TVADD, TVSUB, TVMAC, TVMUL, TVSUM, TVMAX, TVSHUF, TVBROADCAST, TVNEG
    Memory  = 2,  // TLOAD, TSTORE, TVLOAD, TVSTORE
    Branch  = 3,  // TJUMP, TBEQ, TBNE, TBLT, TCALL, TRET
    Control = 4,  // TNOP, THALT, TDBG
};

constexpr int kFuKindCount = 5;

FuKind classify(ter::Opcode op) noexcept;

// Pipeline configuration. G3 adds RAW hazard tracking + forwarding;
// cache/BP land in G4/G5.
struct PipelineConfig {
    int depth        = 7;   // IF, ID, RR, DISP, EX1, EX2/MEM, WB
    int issue_width  = 1;   // G2: single-issue; G3b will raise to 2.

    // Extra EX latency on top of base 1c (total EX cycles = 1 + extra).
    int tvmac_extra_latency = 1;   // total 2 EX cycles
    int vmul_extra_latency  = 1;
    int mem_extra_latency   = 2;   // L1D 3c (G4 replaces this with cache model)

    // G3: RAW dependency tracking.
    bool track_raw          = true;
    // Forwarding: result available at end of last EX (with forwarding) vs end
    // of WB (without). Disable to measure raw stall penalty.
    bool forwarding_enabled = true;
};

struct PipelineReport {
    std::uint64_t cycles_total       = 0;
    std::uint64_t insns_total        = 0;
    std::uint64_t mshr_stall_cycles  = 0;
    std::uint64_t bp_penalty_cycles  = 0;
    std::array<std::uint64_t, kFuKindCount> insns_by_fu{};

    double ipc() const noexcept {
        return cycles_total ? static_cast<double>(insns_total) / cycles_total : 0.0;
    }
};

// Register file classes for dependency tracking.
enum class RegFileKind : std::uint8_t { None, Scalar, Vector, Acc };

// Per-opcode operand info (which file each operand lives in, RMW flag for ops
// that read their dst before writing — TVMAC's accumulator).
struct OpInfo {
    RegFileKind src1 = RegFileKind::None;
    RegFileKind src2 = RegFileKind::None;
    RegFileKind dst  = RegFileKind::None;
    bool        rmw_dst = false;   // reads dst as input (accumulator-style)
};

OpInfo op_info(ter::Opcode op) noexcept;

// Single-issue in-order with RAW hazard tracking + EX-bypass forwarding.
// Streaming: feed traces one at a time via on_retire(); no buffering.
class PipelineModel {
public:
    explicit PipelineModel(PipelineConfig cfg = {}) noexcept : cfg_(cfg) {}

    // Optional data-side cache hierarchy. When set, memory-op latency is
    // computed dynamically (l1d hit / l2 hit / DRAM) instead of the static
    // cfg_.mem_extra_latency. InstrTrace::mem_addr must be populated.
    void set_mem_hierarchy(MemHierarchy* h) noexcept { mem_hier_ = h; }

    // Optional branch predictor. When set, branch ops add mispredict penalty
    // to next_issue_cycle on misses. InstrTrace::branch_taken / branch_target
    // must be populated for branch ops.
    void set_branch_predictor(BranchPredictor* bp) noexcept { bp_ = bp; }

    void on_retire(InstrTrace& t) noexcept;

    const PipelineReport& report() const noexcept { return report_; }
    const PipelineConfig& config() const noexcept { return cfg_; }

private:
    int extra_latency(ter::Opcode op) const noexcept;
    std::uint64_t  ready_cycle(RegFileKind file, std::uint8_t idx) const noexcept;
    void           set_ready(RegFileKind file, std::uint8_t idx, std::uint64_t c) noexcept;

    PipelineConfig    cfg_;
    MemHierarchy*     mem_hier_         = nullptr;
    BranchPredictor*  bp_               = nullptr;
    std::uint64_t     next_issue_cycle_ = 0;
    std::uint64_t  last_retire_cycle_ = 0;
    PipelineReport report_;

    // ready_cycle_[file][idx] = cycle at which the value in (file, idx) is
    // visible to a consumer's read stage. Bumped on every write.
    // Sizes: scalar 27 (R0..R26), vector 9 (V0..V8), acc 3 (A0..A2).
    std::array<std::uint64_t, 27> ready_scalar_{};
    std::array<std::uint64_t, 9>  ready_vector_{};
    std::array<std::uint64_t, 3>  ready_acc_{};
};

}  // namespace tersim
