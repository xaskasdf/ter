#include <ter/host/load_gguf.hpp>
#include <ter/tfloat.hpp>
#include <ter/bitnet.hpp>
#include <core/dequant.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ter::host {

ter::TritTensor tensor_to_trit(const nt::Tensor& t, int n_trits_per_elem,
                               bool format_a_roundtrip, int format_a_mant_trits,
                               bool bitnet_roundtrip, int block_size,
                               bool transpose_2d, bool store_fp32) {
    if (t.device() != nt::Device::CPU) {
        throw std::runtime_error("tensor_to_trit: input must be on CPU");
    }

    // Compute element count from shape.
    // nt::Tensor::shape() returns const std::vector<int64_t>&
    std::size_t n_elems = 1;
    std::vector<int> shape_int;
    for (auto d : t.shape()) {
        n_elems *= static_cast<std::size_t>(d);
        shape_int.push_back(static_cast<int>(d));
    }

    // Dequantize to a temporary float buffer.
    std::vector<float> tmp(n_elems);
    switch (t.dtype()) {
        case nt::DType::F32:
            nt::dequant_f32(t.data(), n_elems, tmp.data());
            break;
        case nt::DType::F16:
            nt::dequant_f16(t.data(), n_elems, tmp.data());
            break;
        case nt::DType::Q8_0:
            nt::dequant_q8_0(t.data(), n_elems, tmp.data());
            break;
        case nt::DType::Q4_K_M:
            nt::dequant_q4_k_m(t.data(), n_elems, tmp.data());
            break;
        case nt::DType::Q6_K:
            nt::dequant_q6_k(t.data(), n_elems, tmp.data());
            break;
        case nt::DType::I2_S:
            nt::dequant_i2_s(t.data(), n_elems, tmp.data());
            break;
        default:
            throw std::runtime_error("tensor_to_trit: unsupported dtype "
                                     "(F16/F32/Q8_0/Q4_K_M/Q6_K/I2_S supported; Q5_K/Q2_K land later)");
    }

    // Optional Format A round-trip: bake tfloat encoding noise into the float
    // buffer before Format B quantization.
    if (format_a_roundtrip) {
        for (std::size_t i = 0; i < n_elems; ++i) {
            tmp[i] = ter::TFloat::from_float_trits(tmp[i], format_a_mant_trits).to_float();
        }
    }

    // Optional BitNet b1.58 round-trip: per-tensor absmean scale, clamp to
    // {-1, 0, +1}, then expand back to float for Format B encoding. This
    // simulates "what if the model had ternary weights" without changing the
    // forward path. Combined with n_trits=9 the payload retains values in
    // {-mti, 0, +mti}; the matmul math is identical -- only the underlying
    // value distribution shifts.
    if (bitnet_roundtrip) {
        std::vector<int8_t> bn(n_elems);
        float scale = ter::quantize_bitnet(tmp.data(), n_elems, bn.data());
        for (std::size_t i = 0; i < n_elems; ++i) {
            tmp[i] = static_cast<float>(bn[i]) * scale;
        }
    }

    // GGUF/ggml stores a linear weight as [out][in] row-major (ne0=in fastest,
    // ne1=out). mm_row expects [in][out] (Wp[k*N+j], k=in, j=out). Transpose
    // here so the projection weights match mm_row's contract. token_embd /
    // lm_head are consumed as [out][in] directly (embed gather + lm_head
    // double-loop), so they pass transpose_2d=false.
    if (transpose_2d && shape_int.size() == 2) {
        const int d0 = shape_int[0];   // ne0 = in  = K
        const int d1 = shape_int[1];   // ne1 = out = N
        std::vector<float> tr(n_elems);
        for (int o = 0; o < d1; ++o)        // out (outer in source [out][in])
            for (int i = 0; i < d0; ++i)    // in
                tr[static_cast<std::size_t>(i) * static_cast<std::size_t>(d1)
                   + static_cast<std::size_t>(o)] =
                    tmp[static_cast<std::size_t>(o) * static_cast<std::size_t>(d0)
                        + static_cast<std::size_t>(i)];
        tmp.swap(tr);
        // Payload is now [in][out] = [K][N] row-major.
    }

    // fp32 reference mode: store the (transposed) float weights directly,
    // skip quantization entirely. Used to validate the forward at true FP.
    if (store_fp32) {
        ter::TritTensor t_out;
        t_out.dtype = ter::DType::TritFP_B;
        t_out.n_trits_per_elem = n_trits_per_elem;
        t_out.shape = shape_int;
        t_out.fp32 = std::move(tmp);
        return t_out;
    }

    // Quantize via ter::quantize() (per-tensor) or quantize_blocked() when a
    // per-block scale is requested.
    if (block_size > 0) {
        return ter::quantize_blocked(tmp.data(), shape_int, n_trits_per_elem, block_size);
    }
    return ter::quantize(tmp.data(), shape_int, n_trits_per_elem);
}

}  // namespace ter::host
