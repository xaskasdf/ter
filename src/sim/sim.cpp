#include <ter/sim.hpp>
#include <stdexcept>

namespace ter {

void Sim::step() {
    if (regs_.halted()) return;
    auto pc = regs_.pc();
    Word27 raw = mem_.load_word(static_cast<size_t>(pc.to_int()));
    regs_.set_pc(Word27::from_int(pc.to_int() + 1));
    Instr i = decode(raw);
    run_one(i);
}

void Sim::run() {
    constexpr int kSafetyLimit = 1'000'000;
    for (int n = 0; n < kSafetyLimit && !regs_.halted(); ++n) step();
    if (!regs_.halted()) throw std::runtime_error("Sim::run safety limit");
}

}
