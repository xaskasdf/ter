#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/pack.hpp>
#include <cstdlib>

using namespace ter;

TEST_CASE("Trit <-> 2-bit code") {
    CHECK(trit_to_code(T_ZERO) == 0b00);
    CHECK(trit_to_code(T_POS)  == 0b01);
    CHECK(trit_to_code(T_NEG)  == 0b10);
    CHECK(code_to_trit(0b00) == T_ZERO);
    CHECK(code_to_trit(0b01) == T_POS);
    CHECK(code_to_trit(0b10) == T_NEG);
}

TEST_CASE("Word27 <-> uint64 round-trip") {
    int64_t cases[] = {0, 1, -1, 1234567890, -1234567890};
    for (int64_t v : cases) {
        if (std::abs(v) > Word27::max_int()) continue;
        Word27 w = Word27::from_int(v);
        uint64_t packed = pack_word27(w);
        Word27 r = unpack_word27(packed);
        CHECK(r == w);
        CHECK(r.to_int() == v);
    }
}

TEST_CASE("Reserved 0b11 decodes as zero") {
    uint64_t bad = 0b11;
    Word27 r = unpack_word27(bad);
    CHECK(r.trit(0) == T_ZERO);
}
