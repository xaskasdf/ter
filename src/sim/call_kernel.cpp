#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <stdexcept>

namespace ter {

KernelId KernelTable::install(Sim& sim, const std::string& name,
                              const std::vector<Word27>& blob) {
    KernelId id;
    id.entry_addr = next_addr_;
    id.valid = true;
    for (size_t i = 0; i < blob.size(); ++i) {
        sim.mem().store_word(next_addr_++, blob[i]);
    }
    by_name_[name] = id;
    return id;
}

KernelId KernelTable::find(const std::string& name) const noexcept {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? KernelId{} : it->second;
}

int64_t Sim::call_kernel(KernelTable&, KernelId id,
                         const std::vector<int64_t>& args) {
    if (!id.valid) throw std::runtime_error("call_kernel: invalid KernelId");
    if (args.size() > 7) throw std::runtime_error("call_kernel: max 7 register args");

    for (size_t i = 0; i < args.size(); ++i) {
        regs_.write_scalar(static_cast<int>(i + 1), Word27::from_int(args[i]));
    }
    regs_.set_pc(Word27::from_int(static_cast<int64_t>(id.entry_addr)));
    regs_.set_halted(false);
    run();
    return regs_.read_scalar(1).to_int();
}

}
