#pragma once
#include <ter/memory.hpp>
#include <ter/regfile.hpp>
#include <ter/counters.hpp>
#include <ter/isa.hpp>

namespace ter {

class Sim {
public:
    explicit Sim(size_t mem_words) : mem_(mem_words) {}

    Memory&           mem()         noexcept { return mem_; }
    const Memory&     mem()   const noexcept { return mem_; }
    RegFile&          regs()        noexcept { return regs_; }
    const RegFile&    regs()  const noexcept { return regs_; }
    OpCounters&       counters()       noexcept { return counters_; }
    const OpCounters& counters() const noexcept { return counters_; }

    void step();
    void run();

private:
    void run_one(const Instr& i);

    Memory mem_;
    RegFile regs_;
    OpCounters counters_;
};

}
