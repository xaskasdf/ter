#pragma once
#include <ter/word.hpp>
#include <stdexcept>
#include <cstdint>

namespace ter {

enum class Opcode : int16_t {
    TNOP    = 0,
    THALT   = 1,
    TDBG    = 2,
    TADD    = 10,
    TSUB    = 11,
    TNEG    = 12,
    TABS    = 13,
    TAND3   = 14,
    TOR3    = 15,
    TXOR3   = 16,
    TCMP    = 17,
    TSIGN   = 18,
    TLOAD   = 30,
    TSTORE  = 31,
    TLOADI  = 32,
    TBEQ    = 50,
    TBNE    = 51,
    TBLT    = 52,
    TJUMP   = 53,
    TCALL   = 54,
    TRET    = 55,
};

struct Instr {
    Opcode op;
    uint8_t dst;
    uint8_t src1;
    uint8_t src2;
    int32_t imm;
};

class IllegalOpcode : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

Word27 encode(const Instr& i);
Instr  decode(Word27 w);

constexpr int kOpcodeTrits = 6;
constexpr int kRegTrits    = 3;
constexpr int kImmTrits    = 12;

}
