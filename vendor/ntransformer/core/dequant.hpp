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

// Q6_K block dequant. ggml format (block_q6_K, 210 bytes / 256 elements):
//   uint8_t     ql[128];          (128 bytes) low 4 bits of quants
//   uint8_t     qh[64];           ( 64 bytes) high 2 bits of quants
//   int8_t      scales[16];       ( 16 bytes) one int8 scale per 16 elems
//   ggml_fp16_t d;                (  2 bytes) super-block scale
// Per element: q6 = (ql_nibble | (qh_pair << 4)) - 32;  y = d * scales[i/16] * q6.
// n_elems must be a multiple of 256.
void dequant_q6_k(const void* src, std::size_t n_elems, float* out);

// Q4_K_M block dequant. ggml format (block_q4_K, 144 bytes / 256 elements):
//   ggml_fp16_t d, dmin;          (4 bytes)   super-block scales
//   uint8_t     scales[12];       (12 bytes)  packed 6-bit (scale,min) for 8 sub-blocks
//   uint8_t     qs[128];          (128 bytes) 4-bit quants (256 elems, 2 per byte)
// Per sub-block j (32 elems): elem = d * scale[j] * q - dmin * min[j].
// n_elems must be a multiple of 256.
void dequant_q4_k_m(const void* src, std::size_t n_elems, float* out);

}  // namespace nt
