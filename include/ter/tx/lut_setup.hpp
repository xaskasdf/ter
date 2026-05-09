#pragma once
#include <ter/sim.hpp>
#include <ter/tx/forward.hpp>
#include <string>

namespace ter::tx {

// Loads rsqrt + sigmoid + exp + rcp LUTs from disk into sim memory at the
// canonical addresses (5000/5300/5600/5900). Returns a LutAddrs struct with
// the four addresses populated. Reads:
//   <lut_dir>/rsqrt_lut.bin
//   <lut_dir>/sigmoid_lut.bin
//   <lut_dir>/exp_lut.bin
//   <lut_dir>/rcp_lut.bin
LutAddrs load_default_luts(Sim& sim, const std::string& lut_dir);

}  // namespace ter::tx
