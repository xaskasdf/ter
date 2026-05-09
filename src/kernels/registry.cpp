#include <ter/kernels.hpp>
#include <ter/sim.hpp>
#include <ter/assembler.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ter {

static std::string read_text(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("registry: cannot open " + path);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

void install_default_kernels(Sim& sim, KernelTable& kt, const std::string& kernels_dir) {
    auto src = read_text(kernels_dir + "/tk_matmul_b_9t.tasm");
    auto blob = assemble(src);
    kt.install(sim, "tk_matmul_b_9t", blob);

    auto src_rms = read_text(kernels_dir + "/tk_rmsnorm.tasm");
    auto blob_rms = assemble(src_rms);
    kt.install(sim, "tk_rmsnorm", blob_rms);
}

}
