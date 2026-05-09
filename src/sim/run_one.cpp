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
        case Opcode::TLOAD: {
            auto addr = regs_.read_scalar(i.src1).to_int();
            regs_.write_scalar(i.dst, mem_.load_word(static_cast<size_t>(addr)));
            break;
        }
        case Opcode::TSTORE: {
            auto addr = regs_.read_scalar(i.src2).to_int();
            auto val  = regs_.read_scalar(i.src1);
            mem_.store_word(static_cast<size_t>(addr), val);
            break;
        }
        case Opcode::TJUMP:
            regs_.set_pc(Word27::from_int(i.imm));
            break;
        case Opcode::TBEQ: {
            auto a = regs_.read_scalar(i.src1);
            auto b = regs_.read_scalar(i.src2);
            if (a == b) regs_.set_pc(Word27::from_int(i.imm));
            break;
        }
        case Opcode::TBNE: {
            auto a = regs_.read_scalar(i.src1);
            auto b = regs_.read_scalar(i.src2);
            if (!(a == b)) regs_.set_pc(Word27::from_int(i.imm));
            break;
        }
        case Opcode::TBLT: {
            auto a = regs_.read_scalar(i.src1);
            auto b = regs_.read_scalar(i.src2);
            if (sign_trit(a - b) == T_NEG) regs_.set_pc(Word27::from_int(i.imm));
            break;
        }
        case Opcode::TCALL: {
            auto sp_w = regs_.read_scalar(26);
            auto sp   = sp_w.to_int();
            mem_.store_word(static_cast<size_t>(sp), regs_.pc());
            regs_.write_scalar(26, Word27::from_int(sp + 1));
            regs_.set_pc(Word27::from_int(i.imm));
            break;
        }
        case Opcode::TRET: {
            auto sp_w = regs_.read_scalar(26);
            auto sp   = sp_w.to_int() - 1;
            regs_.write_scalar(26, Word27::from_int(sp));
            regs_.set_pc(mem_.load_word(static_cast<size_t>(sp)));
            break;
        }
        case Opcode::TAND3: {
            auto a = regs_.read_scalar(i.src1);
            auto b = regs_.read_scalar(i.src2);
            Word27 r;
            for (int k = 0; k < Word27::kTrits; ++k) r.set_trit(k, trit_min(a.trit(k), b.trit(k)));
            regs_.write_scalar(i.dst, r);
            break;
        }
        case Opcode::TOR3: {
            auto a = regs_.read_scalar(i.src1);
            auto b = regs_.read_scalar(i.src2);
            Word27 r;
            for (int k = 0; k < Word27::kTrits; ++k) r.set_trit(k, trit_max(a.trit(k), b.trit(k)));
            regs_.write_scalar(i.dst, r);
            break;
        }
        case Opcode::TXOR3: {
            auto a = regs_.read_scalar(i.src1);
            auto b = regs_.read_scalar(i.src2);
            Word27 r;
            for (int k = 0; k < Word27::kTrits; ++k) {
                if (a.trit(k) == b.trit(k)) r.set_trit(k, T_ZERO);
                else                         r.set_trit(k, -a.trit(k));
            }
            regs_.write_scalar(i.dst, r);
            break;
        }
        case Opcode::TCMP: {
            auto a = regs_.read_scalar(i.src1);
            auto b = regs_.read_scalar(i.src2);
            Word27 r;
            r.set_trit(0, sign_trit(a - b));
            regs_.write_scalar(i.dst, r);
            break;
        }
        case Opcode::TVADD:
            regs_.write_vec(i.dst, vec_add(regs_.read_vec(i.src1), regs_.read_vec(i.src2)));
            break;
        case Opcode::TVSUB:
            regs_.write_vec(i.dst, vec_sub(regs_.read_vec(i.src1), regs_.read_vec(i.src2)));
            break;
        case Opcode::TVNEG:
            regs_.write_vec(i.dst, vec_neg(regs_.read_vec(i.src1)));
            break;
        case Opcode::TVBROADCAST:
            regs_.write_vec(i.dst, vec_broadcast(i.imm));
            break;
        case Opcode::TVMAC:
            vec_mac(regs_.acc(i.dst), regs_.read_vec(i.src1), regs_.read_vec(i.src2));
            break;
        case Opcode::TVSUM: {
            int64_t s_ = vec_sum(regs_.read_acc(i.src1));
            regs_.write_scalar(i.dst, Word27::from_int(s_));
            break;
        }
        case Opcode::TVMAX: {
            auto a = regs_.read_vec(i.src1);
            Vec r = a;
            for (int k = 1; k < Vec::kLanes; ++k) {
                if (r.lane(k - 1) > r.lane(k)) r.set_lane(k, r.lane(k - 1));
            }
            regs_.write_vec(i.dst, r);
            break;
        }
        case Opcode::TVSHUF: {
            auto a = regs_.read_vec(i.src1);
            Vec r;
            int n = Vec::kLanes;
            int shift = ((i.imm % n) + n) % n;
            for (int k = 0; k < n; ++k) r.set_lane(k, a.lane((k + shift) % n));
            regs_.write_vec(i.dst, r);
            break;
        }
        case Opcode::TVLOAD: {
            auto base = regs_.read_scalar(i.src1).to_int();
            Vec r;
            for (int k = 0; k < Vec::kLanes; ++k)
                r.set_lane(k, static_cast<int32_t>(mem_.load_word(static_cast<size_t>(base + k)).to_int()));
            regs_.write_vec(i.dst, r);
            break;
        }
        case Opcode::TVSTORE: {
            auto base = regs_.read_scalar(i.src2).to_int();
            auto v    = regs_.read_vec(i.src1);
            for (int k = 0; k < Vec::kLanes; ++k)
                mem_.store_word(static_cast<size_t>(base + k), Word27::from_int(v.lane(k)));
            break;
        }
        default:
            throw IllegalOpcode("Sim::run_one: opcode not yet implemented");
    }
}

}
