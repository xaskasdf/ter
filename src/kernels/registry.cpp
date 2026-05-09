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

    auto src_sm = read_text(kernels_dir + "/tk_softmax.tasm");
    auto blob_sm = assemble(src_sm);
    kt.install(sim, "tk_softmax", blob_sm);

    auto src_silu = read_text(kernels_dir + "/tk_silu.tasm");
    auto blob_silu = assemble(src_silu);
    kt.install(sim, "tk_silu", blob_silu);

    auto src_rope = read_text(kernels_dir + "/tk_rope.tasm");
    auto blob_rope = assemble(src_rope);
    kt.install(sim, "tk_rope", blob_rope);
}

}
