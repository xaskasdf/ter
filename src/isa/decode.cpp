#include <ter/isa.hpp>
#include <string>

namespace ter {

namespace {
int64_t get_trits(const Word27& w, int start, int n) {
    int64_t acc = 0, place = 1;
    for (int i = 0; i < n; ++i) {
        acc += static_cast<int64_t>(w.trit(start + i).value()) * place;
        place *= 3;
    }
    return acc;
}

bool is_valid_opcode(int64_t v) {
    switch (static_cast<Opcode>(v)) {
        case Opcode::TNOP: case Opcode::THALT: case Opcode::TDBG:
        case Opcode::TADD: case Opcode::TSUB: case Opcode::TNEG: case Opcode::TABS:
        case Opcode::TAND3: case Opcode::TOR3: case Opcode::TXOR3:
        case Opcode::TCMP: case Opcode::TSIGN:
        case Opcode::TLOAD: case Opcode::TSTORE: case Opcode::TLOADI:
        case Opcode::TBEQ: case Opcode::TBNE: case Opcode::TBLT:
        case Opcode::TJUMP: case Opcode::TCALL: case Opcode::TRET:
        case Opcode::TVADD: case Opcode::TVSUB: case Opcode::TVNEG: case Opcode::TVBROADCAST:
        case Opcode::TVMAC: case Opcode::TVSUM: case Opcode::TVMAX: case Opcode::TVSHUF:
        case Opcode::TVLOAD: case Opcode::TVSTORE: case Opcode::TVMUL:
            return true;
    }
    return false;
}
}

Instr decode(Word27 w) {
    int64_t op_v = get_trits(w, 21, kOpcodeTrits);
    if (!is_valid_opcode(op_v)) {
        throw IllegalOpcode("decode: opcode value " + std::to_string(op_v) + " unmapped");
    }
    Instr i;
    i.op   = static_cast<Opcode>(op_v);
    i.dst  = static_cast<uint8_t>(get_trits(w, 18, kRegTrits) + 13);
    i.src1 = static_cast<uint8_t>(get_trits(w, 15, kRegTrits) + 13);
    i.src2 = static_cast<uint8_t>(get_trits(w, 12, kRegTrits) + 13);
    i.imm  = static_cast<int32_t>(get_trits(w, 0, kImmTrits));
    return i;
}

}
