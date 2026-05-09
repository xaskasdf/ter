#include "tensor.h"
#include <numeric>
#include <cstdio>
#include <cstring>
#include <algorithm>

// No CUDA includes — CPU-only vendor build.
// nt_aligned_alloc / nt_aligned_free / NT_CHECK are provided by types.h.

namespace nt {

// ============================================================
// Destructor / Move
// ============================================================

Tensor::~Tensor() {
    free();
}

Tensor::Tensor(Tensor&& other) noexcept
    : name(std::move(other.name))
    , data_(other.data_)
    , shape_(std::move(other.shape_))
    , strides_(std::move(other.strides_))
    , dtype_(other.dtype_)
    , device_(other.device_)
    , is_view_(other.is_view_)
    , owns_data_(other.owns_data_)
    , offset_(other.offset_)
    , data_owner_(std::move(other.data_owner_))
{
    other.data_ = nullptr;
    other.owns_data_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        free();
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        dtype_ = other.dtype_;
        device_ = other.device_;
        is_view_ = other.is_view_;
        owns_data_ = other.owns_data_;
        offset_ = other.offset_;
        data_owner_ = std::move(other.data_owner_);
        name = std::move(other.name);
        other.data_ = nullptr;
        other.owns_data_ = false;
    }
    return *this;
}

// ============================================================
// Factory methods
// ============================================================

Tensor Tensor::empty(const std::vector<int64_t>& shape, DType dtype, Device device) {
    Tensor t;
    t.shape_ = shape;
    t.dtype_ = dtype;
    t.device_ = device;
    t.compute_strides();
    t.allocate();
    t.owns_data_ = true;
    return t;
}

Tensor Tensor::from_ptr(void* data, const std::vector<int64_t>& shape, DType dtype, Device device) {
    Tensor t;
    t.data_ = data;
    t.shape_ = shape;
    t.dtype_ = dtype;
    t.device_ = device;
    t.owns_data_ = false;
    t.is_view_ = true;
    t.compute_strides();
    return t;
}

Tensor Tensor::zeros(const std::vector<int64_t>& shape, DType dtype, Device device) {
    Tensor t = empty(shape, dtype, device);
    size_t bytes = t.nbytes();
    memset(t.data_, 0, bytes);
    return t;
}

// ============================================================
// View operations
// ============================================================

Tensor Tensor::view(const std::vector<int64_t>& new_shape) const {
    // Verify total elements match; support one -1 dimension
    int64_t old_numel = numel();
    int64_t new_numel = 1;
    int infer_dim = -1;
    for (int i = 0; i < (int)new_shape.size(); i++) {
        if (new_shape[i] == -1) {
            NT_CHECK(infer_dim == -1, "Only one dimension can be -1");
            infer_dim = i;
        } else {
            new_numel *= new_shape[i];
        }
    }

    std::vector<int64_t> shape = new_shape;
    if (infer_dim >= 0) {
        shape[infer_dim] = old_numel / new_numel;
        new_numel *= shape[infer_dim];
    }

    NT_CHECK(old_numel == new_numel, "View shape mismatch");

    Tensor t;
    t.shape_ = shape;
    t.dtype_ = dtype_;
    t.device_ = device_;
    t.is_view_ = true;
    t.owns_data_ = false;
    t.data_ = data_;
    t.data_owner_ = data_owner_;
    t.compute_strides();
    return t;
}

Tensor Tensor::slice(int dim, int64_t start, int64_t end) const {
    NT_CHECK(dim >= 0 && dim < ndim(), "Invalid slice dim");
    NT_CHECK(start >= 0 && end <= shape_[dim] && start < end, "Invalid slice range");

    Tensor t;
    t.shape_ = shape_;
    t.shape_[dim] = end - start;
    t.dtype_ = dtype_;
    t.device_ = device_;
    t.is_view_ = true;
    t.owns_data_ = false;
    t.strides_ = strides_;
    t.data_owner_ = data_owner_;

    // Compute byte offset
    size_t elem_size = dtype_size(dtype_);
    if (dtype_block_size(dtype_) > 1) {
        // For quantized types, slicing along the innermost dim must be block-aligned
        if (dim == ndim() - 1) {
            size_t bs = dtype_block_size(dtype_);
            NT_CHECK(start % bs == 0, "Quantized slice must be block-aligned");
            t.offset_ = offset_ + (start / bs) * elem_size;
        } else {
            t.offset_ = offset_ + start * strides_[dim] * elem_size;
        }
    } else {
        t.offset_ = offset_ + start * strides_[dim] * elem_size;
    }
    t.data_ = static_cast<char*>(const_cast<void*>(data_)) + t.offset_ - offset_;

    return t;
}

// ============================================================
// Metadata
// ============================================================

int64_t Tensor::numel() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), (int64_t)1, std::multiplies<int64_t>());
}

size_t Tensor::nbytes() const {
    int64_t n = numel();
    size_t bs = dtype_block_size(dtype_);
    if (bs > 1) {
        return (n / bs) * dtype_size(dtype_);
    }
    return static_cast<size_t>(n) * dtype_size(dtype_);
}

bool Tensor::is_contiguous() const {
    if (shape_.empty()) return true;
    int64_t expected = 1;
    for (int i = ndim() - 1; i >= 0; i--) {
        if (strides_[i] != expected) return false;
        expected *= shape_[i];
    }
    return true;
}

// ============================================================
// Device transfer (CPU-only)
// ============================================================

Tensor Tensor::to(Device target) const {
    // CPU-only: only Device::CPU is valid
    (void)target;  // target must be CPU
    // Same device — create a copy
    Tensor t = Tensor::empty(shape_, dtype_, device_);
    size_t bytes = nbytes();
    memcpy(t.data_, data_, bytes);
    return t;
}

void Tensor::copy_from(const Tensor& src) {
    NT_CHECK(numel() == src.numel(), "copy_from: size mismatch");
    NT_CHECK(dtype_ == src.dtype_, "copy_from: dtype mismatch");
    size_t bytes = nbytes();
    memcpy(data_, src.data_, bytes);
}

// ============================================================
// Debug
// ============================================================

std::string Tensor::to_string() const {
    std::string s = "Tensor(";
    s += "shape=[";
    for (int i = 0; i < ndim(); i++) {
        if (i > 0) s += ", ";
        s += std::to_string(shape_[i]);
    }
    s += "], dtype=" + std::string(dtype_name(dtype_));
    s += ", device=CPU";
    if (!name.empty()) s += ", name=" + name;
    s += ")";
    return s;
}

void Tensor::print_info() const {
    fprintf(stderr, "%s\n", to_string().c_str());
}

// ============================================================
// Private helpers
// ============================================================

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;
    strides_.back() = 1;
    for (int i = (int)shape_.size() - 2; i >= 0; i--) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

void Tensor::allocate() {
    size_t bytes = nbytes();
    if (bytes == 0) return;

    // CPU-only: use aligned allocation for SIMD
    data_ = nt_aligned_alloc(64, (bytes + 63) & ~63);
    NT_CHECK(data_ != nullptr, "CPU allocation failed");

    // Set up shared ownership for views
    void* raw = data_;
    data_owner_ = std::shared_ptr<void>(raw, [](void* p) {
        nt_aligned_free(p);
    });
    owns_data_ = true;
}

void Tensor::free() {
    if (owns_data_ && data_ && !data_owner_) {
        nt_aligned_free(data_);
    }
    data_ = nullptr;
    owns_data_ = false;
    data_owner_.reset();
}

} // namespace nt
