#pragma once
#include <ter/isa.hpp>
#include <cstdint>

namespace tersim {

// One retired instruction as seen by the microarchitectural model.
// Filled by Sim's tracer callback (functional level: pc, op, operands)
// and later annotated by the pipeline (cycle_issued, cycle_retired,
// stall reasons). Streamed — never persisted in full for production runs;
// a BitNet 2B-4T forward generates ~10^10 entries.
struct InstrTrace {
    std::uint64_t pc          = 0;
    ter::Opcode   op          = ter::Opcode{};
    std::uint8_t  dst         = 0;
    std::uint8_t  src1        = 0;
    std::uint8_t  src2        = 0;
    std::int32_t  imm         = 0;
    std::uint64_t mem_addr    = 0;       // valid iff op touches memory
    bool          mem_is_load = false;
    bool          mem_is_store = false;

    // Branch info (populated by tracer for branch ops; ignored otherwise).
    bool          branch_taken  = false;
    std::uint64_t branch_target = 0;

    // Filled by Layer-1 pipeline model:
    std::uint64_t cycle_issued  = 0;
    std::uint64_t cycle_retired = 0;
};

}  // namespace tersim
