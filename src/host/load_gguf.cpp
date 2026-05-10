#include <ter/host/load_gguf.hpp>
#include <core/dequant.hpp>
#include <stdexcept>
#include <vector>

namespace ter::host {

ter::TritTensor tensor_to_trit(const nt::Tensor& t, int n_trits_per_elem) {
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
        default:
            throw std::runtime_error("tensor_to_trit: unsupported dtype "
                                     "(F16/F32/Q8_0/Q4_K_M/Q6_K supported; Q5_K/Q2_K land later)");
    }

    // Quantize via ter::quantize().
    return ter::quantize(tmp.data(), shape_int, n_trits_per_elem);
}

}  // namespace ter::host
