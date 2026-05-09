#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/regfile.hpp>

using namespace ter;

TEST_CASE("R0 is hard-zero") {
    RegFile rf;
    rf.write_scalar(0, Word27::from_int(42));
    CHECK(rf.read_scalar(0).to_int() == 0);
}

TEST_CASE("read/write scalar") {
    RegFile rf;
    rf.write_scalar(1, Word27::from_int(123));
    rf.write_scalar(26, Word27::from_int(-456));
    CHECK(rf.read_scalar(1).to_int() == 123);
    CHECK(rf.read_scalar(26).to_int() == -456);
}

TEST_CASE("PC and halted flag") {
    RegFile rf;
    CHECK(rf.pc().to_int() == 0);
    CHECK_FALSE(rf.halted());
    rf.set_pc(Word27::from_int(100));
    CHECK(rf.pc().to_int() == 100);
    rf.set_halted(true);
    CHECK(rf.halted());
}
