#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <core/tensor.h>
#include <vector>

using namespace nt;

TEST_CASE("Tensor::empty allocates with the requested shape and dtype") {
    auto t = Tensor::empty({4, 8}, DType::F32, Device::CPU);
    CHECK(t.dtype() == DType::F32);
    CHECK(t.device() == Device::CPU);
    CHECK(t.shape().size() == 2);
    CHECK(t.shape()[0] == 4);
    CHECK(t.shape()[1] == 8);
}

TEST_CASE("Tensor::zeros initialises bytes to zero") {
    auto t = Tensor::zeros({16}, DType::F32, Device::CPU);
    auto* p = t.data_as<float>();
    for (int i = 0; i < 16; ++i) CHECK(p[i] == 0.0f);
}

TEST_CASE("Tensor::from_ptr wraps without owning") {
    std::vector<float> buf = {1.0f, 2.0f, 3.0f, 4.0f};
    auto t = Tensor::from_ptr(buf.data(), {4}, DType::F32, Device::CPU);
    CHECK(t.data_as<float>()[2] == 3.0f);
}

TEST_CASE("Tensor::numel and nbytes are correct") {
    auto t = Tensor::empty({3, 4}, DType::F32, Device::CPU);
    CHECK(t.numel() == 12);
    CHECK(t.nbytes() == 12 * 4);
}

TEST_CASE("Tensor move semantics transfer ownership") {
    auto t1 = Tensor::zeros({8}, DType::F32, Device::CPU);
    void* ptr = t1.data();
    auto t2 = std::move(t1);
    CHECK(t2.data() == ptr);
    CHECK(t1.data() == nullptr);  // NOLINT(bugprone-use-after-move)
}

TEST_CASE("Tensor::view reshapes without copy") {
    auto t = Tensor::zeros({4, 4}, DType::F32, Device::CPU);
    auto v = t.view({16});
    CHECK(v.numel() == 16);
    CHECK(v.ndim() == 1);
    CHECK(v.data() == t.data());  // same buffer
}

TEST_CASE("Tensor::is_contiguous is true for freshly allocated tensor") {
    auto t = Tensor::empty({2, 3, 4}, DType::F32, Device::CPU);
    CHECK(t.is_contiguous());
}

TEST_CASE("Tensor::to returns a copy on CPU") {
    auto t = Tensor::zeros({4}, DType::F32, Device::CPU);
    t.data_as<float>()[0] = 42.0f;
    auto t2 = t.to(Device::CPU);
    CHECK(t2.data_as<float>()[0] == 42.0f);
    CHECK(t2.data() != t.data());  // different buffer
}

TEST_CASE("Tensor::copy_from copies data between CPU tensors") {
    auto src = Tensor::zeros({4}, DType::F32, Device::CPU);
    src.data_as<float>()[1] = 7.0f;
    auto dst = Tensor::empty({4}, DType::F32, Device::CPU);
    dst.copy_from(src);
    CHECK(dst.data_as<float>()[1] == 7.0f);
}
