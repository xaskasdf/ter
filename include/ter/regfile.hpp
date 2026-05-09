#pragma once
#include <ter/word.hpp>
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

private:
    static void check(int idx) {
        if (idx < 0 || idx >= kScalarRegs) throw std::out_of_range("RegFile index");
    }

    std::array<Word27, kScalarRegs> scalars_{};
    Word27 pc_{};
    bool halted_ = false;
};

}
