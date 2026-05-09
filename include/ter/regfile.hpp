#pragma once
#include <ter/word.hpp>
#include <ter/vec.hpp>
#include <array>
#include <stdexcept>

namespace ter {

class RegFile {
public:
    static constexpr int kScalarRegs = 27;

    Word27 read_scalar(int idx) const {
        check(idx);
        if (idx == 0) return Word27{};
        return scalars_[idx];
    }

    void write_scalar(int idx, Word27 v) {
        check(idx);
        if (idx == 0) return;
        scalars_[idx] = v;
    }

    Word27 pc() const noexcept { return pc_; }
    void   set_pc(Word27 v) noexcept { pc_ = v; }

    bool halted() const noexcept { return halted_; }
    void set_halted(bool v) noexcept { halted_ = v; }

    // Zero out caller-saved registers (R8..R15, all V/A). Callee-saved (R16..R25)
    // and PC/halted are untouched.
    void reset_caller_saved() noexcept {
        for (int i = 8; i <= 15; ++i) scalars_[i] = Word27{};
        for (int i = 0; i < kVecRegs; ++i) vecs_[i] = Vec{};
        for (int i = 0; i < kAccRegs; ++i) accs_[i] = VAccum{};
    }

    static constexpr int kVecRegs = 9;
    static constexpr int kAccRegs = 3;

    Vec     read_vec(int idx) const     { vcheck(idx); return vecs_[idx]; }
    void    write_vec(int idx, Vec v)   { vcheck(idx); vecs_[idx] = v; }
    VAccum  read_acc(int idx) const     { acheck(idx); return accs_[idx]; }
    VAccum& acc(int idx)                { acheck(idx); return accs_[idx]; }

private:
    static void check(int idx) {
        if (idx < 0 || idx >= kScalarRegs) throw std::out_of_range("RegFile index");
    }
    static void vcheck(int idx) { if (idx<0||idx>=kVecRegs) throw std::out_of_range("vreg"); }
    static void acheck(int idx) { if (idx<0||idx>=kAccRegs) throw std::out_of_range("acc"); }

    std::array<Word27, kScalarRegs> scalars_{};
    Word27 pc_{};
    bool halted_ = false;
    std::array<Vec, kVecRegs>    vecs_{};
    std::array<VAccum, kAccRegs> accs_{};
};

}
