#include <ter/sim.hpp>

namespace ter {

void Sim::run_one(const Instr& i) {
    counters_.bump(i.op);
    switch (i.op) {
        case Opcode::TNOP:
            break;
        case Opcode::THALT:
            regs_.set_halted(true);
            break;
        case Opcode::TDBG:
            break;
        case Opcode::TLOADI:
            regs_.write_scalar(i.dst, Word27::from_int(i.imm));
            break;
        case Opcode::TADD: {
            auto a = regs_.read_scalar(i.src1);
            auto b = regs_.read_scalar(i.src2);
            regs_.write_scalar(i.dst, a + b);
            break;
        }
        case Opcode::TSUB: {
            auto a = regs_.read_scalar(i.src1);
            auto b = regs_.read_scalar(i.src2);
            regs_.write_scalar(i.dst, a - b);
            break;
        }
        case Opcode::TNEG:
            regs_.write_scalar(i.dst, -regs_.read_scalar(i.src1));
            break;
        case Opcode::TABS: {
            auto v = regs_.read_scalar(i.src1);
            regs_.write_scalar(i.dst, sign_trit(v) == T_NEG ? -v : v);
            break;
        }
        case Opcode::TSIGN: {
            auto v = regs_.read_scalar(i.src1);
            Word27 r;
            r.set_trit(0, sign_trit(v));
            regs_.write_scalar(i.dst, r);
            break;
        }
        default:
            throw IllegalOpcode("Sim::run_one: opcode not yet implemented");
    }
}

}
