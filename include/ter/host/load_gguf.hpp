#pragma once
#include <core/tensor.h>
#include <ter/numfmt.hpp>

namespace ter::host {

// Convert an nt::Tensor (Device::CPU; F32, F16, Q8_0, Q4_K_M, Q6_K supported)
// into a Format B TritTensor with the given trit width.
// If format_a_roundtrip is true, the float buffer is rewritten through a
// TFloat encode/decode pair before Format B quantization; this folds Format A
// (1t sign + 5t exp + 9t mantissa) encoding noise into the weights without
// changing the kernel path, isolating "encoding precision impact" as a
// single experimental knob.
// block_size > 0 selects per-block (Q8_0-style) quantization instead of the
// default per-tensor scale — far less lossy for weights with outliers.
// transpose_2d transposes a 2D weight from ggml [out][in] to [in][out] so it
// matches mm_row's Wp[k*N+j] contract (use for projection weights; leave false
// for token_embd / lm_head which are consumed as [out][in]).
// store_fp32 keeps the (optionally transposed) float weights unquantized in
// TritTensor::fp32 — a true-FP reference mode for forward validation.
ter::TritTensor tensor_to_trit(const nt::Tensor& t,
                               int n_trits_per_elem = 9,
                               bool format_a_roundtrip = false,
                               int format_a_mant_trits = 9,
                               bool bitnet_roundtrip = false,
                               int block_size = 0,
                               bool transpose_2d = false,
                               bool store_fp32 = false);

}  // namespace ter::host
