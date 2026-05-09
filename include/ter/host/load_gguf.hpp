#pragma once
#include <core/tensor.h>
#include <ter/numfmt.hpp>

namespace ter::host {

// Convert an nt::Tensor (Device::CPU, dtype = F32 or F16) into a Format B
// TritTensor with the given trit width. Throws std::runtime_error for any
// other dtype (Q4_0 / Q8_0 / etc. land in F5.4c with proper unpackers).
ter::TritTensor tensor_to_trit(const nt::Tensor& t, int n_trits_per_elem = 9);

}  // namespace ter::host
