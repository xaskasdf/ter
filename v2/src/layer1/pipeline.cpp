#include <tersim/cycle_model.hpp>
#include <tersim/cache_model.hpp>
#include <tersim/bp_model.hpp>
#include <algorithm>

namespace tersim {

namespace {
bool is_mem_op(ter::Opcode op) noexcept {
    return op == ter::Opcode::TLOAD  || op == ter::Opcode::TSTORE
        || op == ter::Opcode::TVLOAD || op == ter::Opcode::TVSTORE;
}
bool is_load(ter::Opcode op) noexcept {
    return op == ter::Opcode::TLOAD || op == ter::Opcode::TVLOAD;
}
bool is_branch_op(ter::Opcode op) noexcept {
    return op == ter::Opcode::TBEQ  || op == ter::Opcode::TBNE
        || op == ter::Opcode::TBLT  || op == ter::Opcode::TJUMP
        || op == ter::Opcode::TCALL || op == ter::Opcode::TRET;
}
}  // namespace

FuKind classify(ter::Opcode op) noexcept {
    using O = ter::Opcode;
    switch (op) {
        case O::TVADD: case O::TVSUB: case O::TVNEG: case O::TVBROADCAST:
        case O::TVMAC: case O::TVSUM: case O::TVMAX: case O::TVSHUF:
        case O::TVMUL:
            return FuKind::Vector;

        case O::TLOAD: case O::TSTORE: case O::TVLOAD: case O::TVSTORE:
            return FuKind::Memory;

        case O::TJUMP: case O::TBEQ: case O::TBNE: case O::TBLT:
        case O::TCALL: case O::TRET:
            return FuKind::Branch;

        case O::TNOP: case O::THALT: case O::TDBG:
            return FuKind::Control;

        default:
            return FuKind::Scalar;
    }
}

OpInfo op_info(ter::Opcode op) noexcept {
    using O = ter::Opcode;
    using R = RegFileKind;
    OpInfo i;
    switch (op) {
        // Scalar 3-operand arithmetic / logical: dst = f(src1, src2).
        case O::TADD: case O::TSUB:
        case O::TAND3: case O::TOR3: case O::TXOR3: case O::TCMP:
            i.src1 = R::Scalar; i.src2 = R::Scalar; i.dst = R::Scalar; break;

        // Scalar unary.
        case O::TNEG: case O::TABS: case O::TSIGN:
            i.src1 = R::Scalar; i.dst = R::Scalar; break;

        // Scalar load-immediate: dst only.
        case O::TLOADI:
            i.dst = R::Scalar; break;

        // Memory.
        case O::TLOAD:    // dst = mem[src1]
            i.src1 = R::Scalar; i.dst = R::Scalar; break;
        case O::TSTORE:   // mem[src2] = src1
            i.src1 = R::Scalar; i.src2 = R::Scalar; break;
        case O::TVLOAD:   // dst_vec = mem[src1_scalar]
            i.src1 = R::Scalar; i.dst = R::Vector; break;
        case O::TVSTORE:  // mem[src2_scalar] = src1_vec
            i.src1 = R::Vector; i.src2 = R::Scalar; break;

        // Branches read two scalar regs, write none (architectural; pc handled
        // implicitly by the branch predictor model in G5).
        case O::TBEQ: case O::TBNE: case O::TBLT:
            i.src1 = R::Scalar; i.src2 = R::Scalar; break;

        // TJUMP: imm only.
        case O::TJUMP: break;

        // Stack ops: read/write R26 (sp). Model as scalar RMW on src1.
        case O::TCALL:
            i.dst = R::Scalar; i.rmw_dst = true; break;
        case O::TRET:
            i.dst = R::Scalar; i.rmw_dst = true; break;

        // Vector arithmetic.
        case O::TVADD: case O::TVSUB:
            i.src1 = R::Vector; i.src2 = R::Vector; i.dst = R::Vector; break;
        case O::TVNEG: case O::TVMAX: case O::TVSHUF:
            i.src1 = R::Vector; i.dst = R::Vector; break;
        case O::TVBROADCAST:
            i.dst = R::Vector; break;
        case O::TVMUL:
            i.src1 = R::Vector; i.src2 = R::Vector; i.dst = R::Vector; break;

        // Accumulator ops.
        case O::TVMAC:        // a[dst] += v[src1] * v[src2]  (RMW on Acc)
            i.src1 = R::Vector; i.src2 = R::Vector;
            i.dst  = R::Acc;    i.rmw_dst = true; break;
        case O::TVSUM:        // dst_scalar = sum(a[src1])
            i.src1 = R::Acc; i.dst = R::Scalar; break;

        // Control flow / noops: no reg traffic.
        case O::TNOP: case O::THALT: case O::TDBG:
        default:
            break;
    }
    return i;
}

int PipelineModel::extra_latency(ter::Opcode op) const noexcept {
    switch (op) {
        case ter::Opcode::TVMAC:                          return cfg_.tvmac_extra_latency;
        case ter::Opcode::TVMUL:                          return cfg_.vmul_extra_latency;
        case ter::Opcode::TLOAD:  case ter::Opcode::TVLOAD:
        case ter::Opcode::TSTORE: case ter::Opcode::TVSTORE:
            return cfg_.mem_extra_latency;
        default:                                          return 0;
    }
}

std::uint64_t PipelineModel::ready_cycle(RegFileKind file, std::uint8_t idx) const noexcept {
    switch (file) {
        case RegFileKind::Scalar: return idx < ready_scalar_.size() ? ready_scalar_[idx] : 0;
        case RegFileKind::Vector: return idx < ready_vector_.size() ? ready_vector_[idx] : 0;
        case RegFileKind::Acc:    return idx < ready_acc_.size()    ? ready_acc_[idx]    : 0;
        default:                  return 0;
    }
}

void PipelineModel::set_ready(RegFileKind file, std::uint8_t idx, std::uint64_t c) noexcept {
    switch (file) {
        case RegFileKind::Scalar: if (idx < ready_scalar_.size()) ready_scalar_[idx] = c; break;
        case RegFileKind::Vector: if (idx < ready_vector_.size()) ready_vector_[idx] = c; break;
        case RegFileKind::Acc:    if (idx < ready_acc_.size())    ready_acc_[idx]    = c; break;
        default: break;
    }
}

void PipelineModel::on_retire(InstrTrace& t) noexcept {
    // Stage layout (depth=7): IF=0, ID=1, RR=2, DISP=3, EX1=4, EX2/MEM=5, WB=6.
    // Read stage:  RR  (issue+2) without forwarding; EX1 (issue+4) with.
    // Write stage: WB  (issue+depth-1+extra)  without forwarding;
    //              end of last EX (issue+4+extra) with forwarding.
    const int read_offset_in_pipe  = cfg_.forwarding_enabled ? 4 : 2;
    int extra                      = extra_latency(t.op);

    // Resolve RAW: for each source, earliest issue such that read stage
    // observes a ready value: issue + read_offset >= producer_ready_cycle.
    std::uint64_t dep_min_issue = 0;
    if (cfg_.track_raw) {
        const auto info = op_info(t.op);
        auto bump = [&](RegFileKind f, std::uint8_t idx) {
            const auto r = ready_cycle(f, idx);
            if (r > static_cast<std::uint64_t>(read_offset_in_pipe)) {
                const auto cand = r - static_cast<std::uint64_t>(read_offset_in_pipe);
                if (cand > dep_min_issue) dep_min_issue = cand;
            }
        };
        if (info.src1 != RegFileKind::None) bump(info.src1, t.src1);
        if (info.src2 != RegFileKind::None) bump(info.src2, t.src2);
        if (info.rmw_dst && info.dst != RegFileKind::None) bump(info.dst, t.dst);
    }

    auto issued = std::max(next_issue_cycle_, dep_min_issue);

    // G5: MSHR-aware memory access. May push issue forward if all MSHRs busy.
    if (mem_hier_ && is_mem_op(t.op)) {
        const auto pre = mem_hier_->stats().mshr_stall_cycles;
        const auto r = mem_hier_->access_at(t.mem_addr, is_load(t.op), issued);
        issued = r.actual_issue_cycle;
        extra  = r.latency_cycles - 1;
        if (extra < 0) extra = 0;
        const auto post = mem_hier_->stats().mshr_stall_cycles;
        report_.mshr_stall_cycles += (post - pre);
    }

    const int write_offset_in_pipe = cfg_.forwarding_enabled
                                         ? (4 + extra)
                                         : (cfg_.depth - 1 + extra);

    const auto retired = issued + static_cast<std::uint64_t>(cfg_.depth - 1 + extra);

    t.cycle_issued  = issued;
    t.cycle_retired = retired;

    // Update destination ready cycle.
    if (cfg_.track_raw) {
        const auto info = op_info(t.op);
        if (info.dst != RegFileKind::None) {
            const auto write_cycle = issued + static_cast<std::uint64_t>(write_offset_in_pipe);
            set_ready(info.dst, t.dst, write_cycle);
        }
    }

    next_issue_cycle_ = issued + 1;

    // G5: branch predictor — on mispredict, squash wrong-path stages by pushing
    // next issue forward by the mispredict penalty.
    if (bp_ && is_branch_op(t.op)) {
        const bool correct = bp_->predict_and_update(t.pc, t.branch_taken, t.branch_target);
        if (!correct) {
            const auto pen = static_cast<std::uint64_t>(bp_->mispredict_penalty());
            next_issue_cycle_ += pen;
            report_.bp_penalty_cycles += pen;
        }
    }

    if (retired > last_retire_cycle_) last_retire_cycle_ = retired;

    ++report_.insns_total;
    report_.insns_by_fu[static_cast<std::size_t>(classify(t.op))]++;
    report_.cycles_total = last_retire_cycle_ + 1;
}

}  // namespace tersim
