#include <ter/tx/lut_setup.hpp>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace ter::tx {

namespace {

std::vector<int> read_int32_lut(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("LUT not found: " + path);
    f.seekg(0, std::ios::end);
    size_t n = static_cast<size_t>(f.tellg()) / 4;
    f.seekg(0);
    std::vector<int32_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4));
    return std::vector<int>(v.begin(), v.end());
}

}  // namespace

LutAddrs load_default_luts(Sim& sim, const std::string& lut_dir) {
    LutAddrs L;
    L.rsqrt   = 5000;
    L.sigmoid = 5300;
    L.exp     = 5600;
    L.rcp     = 5900;

    auto rsqrt   = read_int32_lut(lut_dir + "/rsqrt_lut.bin");
    auto sigmoid = read_int32_lut(lut_dir + "/sigmoid_lut.bin");
    auto exp_l   = read_int32_lut(lut_dir + "/exp_lut.bin");
    auto rcp_l   = read_int32_lut(lut_dir + "/rcp_lut.bin");

    sim.load_lut(static_cast<size_t>(L.rsqrt),   rsqrt);
    sim.load_lut(static_cast<size_t>(L.sigmoid), sigmoid);
    sim.load_lut(static_cast<size_t>(L.exp),     exp_l);
    sim.load_lut(static_cast<size_t>(L.rcp),     rcp_l);
    return L;
}

}  // namespace ter::tx
