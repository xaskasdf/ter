// F10: smoke test for the libter_k4 C API. Confirms create/destroy + counter
// snapshot work; full kernel install/call exercised once the freestanding port
// lands.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter_k4/ter_k4.h>

TEST_CASE("ter_k4 C API: create/destroy + counter snapshot") {
    ter_k4_handle_t* h = ter_k4_create(64 * 1024);
    REQUIRE(h != nullptr);

    ter_k4_op_counts_t c{};
    ter_k4_op_counts(h, &c);
    CHECK(c.total_ops == 0);
    CHECK(c.tvmac == 0);

    ter_k4_reset_counters(h);
    ter_k4_op_counts(h, &c);
    CHECK(c.total_ops == 0);

    // Calling an unknown kernel must return an error, not crash.
    int rc = ter_k4_call(h, "nonexistent_kernel", nullptr, 0);
    CHECK(rc != 0);

    ter_k4_destroy(h);
}
