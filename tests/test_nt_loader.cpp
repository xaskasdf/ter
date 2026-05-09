#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <model/loader.h>

using namespace nt;

TEST_CASE("Loader rejects non-existent file") {
    GGUFLoader l;
    CHECK_FALSE(l.load("/nonexistent/path.gguf"));
}

TEST_CASE("Loader rejects path with bad magic") {
    GGUFLoader l;
    CHECK_FALSE(l.load("/etc/hosts"));
}
