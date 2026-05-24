#pragma once
#include <ter/memory.hpp>
#include <ter/regfile.hpp>
#include <ter/counters.hpp>
#include <ter/isa.hpp>
#include <cstdint>
#include <functional>
#include <vector>

namespace ter {

class Sim {
public:
    // Tracer: optional post-retire callback used by the v2 microarch simulator
    // (tersim) to capture an instruction-level trace without persisting it.
    // Signature: (pc_of_executed_instr, decoded_instr, sim_state_after_exec).
    // Default-empty std::function — zero cost when unset.
    using Tracer = std::function<void(std::uint64_t, const Instr&, const Sim&)>;

    explicit Sim(size_t mem_words) : mem_(mem_words) {}

    Memory&           mem()         noexcept { return mem_; }
    const Memory&     mem()   const noexcept { return mem_; }
    RegFile&          regs()        noexcept { return regs_; }
    const RegFile&    regs()  const noexcept { return regs_; }
    OpCounters&       counters()       noexcept { return counters_; }
    const OpCounters& counters() const noexcept { return counters_; }

    void set_tracer(Tracer t) noexcept { tracer_ = std::move(t); }
    const Tracer& tracer() const noexcept { return tracer_; }

    void step();
    void run();

    int64_t call_kernel(class KernelTable& kt, struct KernelId id,
                        const std::vector<int64_t>& args);

    void load_lut(size_t addr, const std::vector<int>& values);

private:
    void run_one(const Instr& i);

    Memory mem_;
    RegFile regs_;
    OpCounters counters_;
    Tracer tracer_;
};

}
