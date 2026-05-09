#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <ter/sim.hpp>
#include <ter/assembler.hpp>
#include <fstream>
#include <sstream>

using namespace ter;

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    REQUIRE(f.is_open());
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

TEST_CASE("smoke: sum(1..5) program produces 15") {
    auto source = read_file("examples/factorial.tasm");
    auto blob = assemble(source);
    Sim s(256);
    for (size_t i = 0; i < blob.size(); ++i) s.mem().store_word(i, blob[i]);
    s.run();
    CHECK(s.regs().read_scalar(1).to_int() == 15);
    CHECK(s.counters().get(Opcode::TADD)  == 5);
    CHECK(s.counters().get(Opcode::TSUB)  == 5);
    CHECK(s.counters().get(Opcode::TJUMP) == 5);
}
