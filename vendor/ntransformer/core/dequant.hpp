#pragma once
#include <cstdint>
#include <cstddef>

namespace nt {

// Convert a buffer of n IEEE 754 binary16 (half-precision) values to f32.
// src must point to n * 2 bytes; out must hold n floats.
void dequant_f16(const void* src, std::size_t n_elems, float* out);

// f32 passthrough (identity copy). Convenience for callers that do dispatch
// without special-casing.
void dequant_f32(const void* src, std::size_t n_elems, float* out);

// Q8_0 block dequant. ggml format: each block holds 1 ggml_fp16_t scale (`d`)
// followed by 32 int8 quants `qs[]`. Block size: 34 bytes; 32 elements per block.
// n_elems must be a multiple of 32.
void dequant_q8_0(const void* src, std::size_t n_elems, float* out);

}  // namespace nt
