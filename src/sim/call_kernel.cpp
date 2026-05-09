#include <ter/sim.hpp>
#include <ter/kernels.hpp>
#include <ter/assembler.hpp>
#include <stdexcept>

namespace ter {

// Jump-type opcodes whose imm field is a blob-relative label index that must
// be relocated to an absolute memory address when the kernel is installed.
static bool is_jump_op(Opcode op) noexcept {
    switch (op) {
        case Opcode::TBEQ:
        case Opcode::TBNE:
        case Opcode::TBLT:
        case Opcode::TJUMP:
        case Opcode::TCALL:
            return true;
        default:
            return false;
    }
}

KernelId KernelTable::install(Sim& sim, const std::string& name,
                              const std::vector<Word27>& blob) {
    KernelId id;
    id.entry_addr = next_addr_;
    id.valid = true;
    for (size_t i = 0; i < blob.size(); ++i) {
        Instr instr = decode(blob[i]);
        if (is_jump_op(instr.op)) {
            // imm is a blob-relative index; relocate to absolute address.
            instr.imm = static_cast<int32_t>(id.entry_addr) + instr.imm;
            sim.mem().store_word(next_addr_++, encode(instr));
        } else {
            sim.mem().store_word(next_addr_++, blob[i]);
        }
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

    regs_.reset_caller_saved();

    for (size_t i = 0; i < args.size(); ++i) {
        regs_.write_scalar(static_cast<int>(i + 1), Word27::from_int(args[i]));
    }
    regs_.set_pc(Word27::from_int(static_cast<int64_t>(id.entry_addr)));
    regs_.set_halted(false);
    run();
    return regs_.read_scalar(1).to_int();
}

}
