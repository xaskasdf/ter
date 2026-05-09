#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <core/types.h>

using namespace nt;

TEST_CASE("DType::TERNARY is present") {
    CHECK(static_cast<int>(DType::TERNARY) == 9);
}

TEST_CASE("dtype_size and dtype_block_size for TERNARY are non-zero") {
    CHECK(dtype_size(DType::TERNARY) > 0);
    CHECK(dtype_block_size(DType::TERNARY) > 0);
}

TEST_CASE("standard DType entries unchanged") {
    CHECK(dtype_size(DType::F32) == 4);
    CHECK(dtype_size(DType::F16) == 2);
    CHECK(dtype_block_size(DType::F32) == 1);
}
