#include <ter/isa.hpp>

namespace ter {

namespace {
void put_trits(Word27& w, int start, int n, int64_t value) {
    for (int i = 0; i < n; ++i) {
        int r = static_cast<int>(((value % 3) + 3) % 3);
        int digit = (r == 2) ? -1 : r;
        w.set_trit(start + i, Trit{digit});
        value = (value - digit) / 3;
    }
}
}

Word27 encode(const Instr& i) {
    Word27 w;
    put_trits(w, 0,  kImmTrits,    i.imm);
    put_trits(w, 12, kRegTrits,    static_cast<int64_t>(i.src2) - 13);
    put_trits(w, 15, kRegTrits,    static_cast<int64_t>(i.src1) - 13);
    put_trits(w, 18, kRegTrits,    static_cast<int64_t>(i.dst)  - 13);
    put_trits(w, 21, kOpcodeTrits, static_cast<int64_t>(i.op));
    return w;
}

}
