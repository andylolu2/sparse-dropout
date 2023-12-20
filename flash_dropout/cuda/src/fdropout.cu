#include <cute/tensor.hpp>
//
#include <cutlass/util/device_memory.h>
#include <torch/extension.h>

#include <vector>

#include "matmul.cuh"

namespace ct = cute;

template <typename T, typename Layout>
auto torch_to_ct_2d(torch::Tensor x) {
    auto ptr = ct::make_gmem_ptr(reinterpret_cast<T *>(x.data_ptr()));
    return ct::make_tensor(ptr, ct::make_shape(x.size(0), x.size(1)), Layout{});
}

// C++ interface
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t> forward(
    torch::Tensor A, torch::Tensor B, float p) {
    TORCH_CHECK(A.dtype() == torch::kHalf, "A must be half");
    TORCH_CHECK(B.dtype() == torch::kHalf, "B must be half");
    TORCH_CHECK(A.type().is_cuda(), "A must be CUDA");
    TORCH_CHECK(B.type().is_cuda(), "B must be CUDA");
    TORCH_CHECK(A.ndimension() == 2, "A must be 2D");
    TORCH_CHECK(B.ndimension() == 2, "B must be 2D");
    TORCH_CHECK(B.size(1) == A.size(1), "B must have the same k dim size as A");
    TORCH_CHECK(A.stride(0) == A.size(1) && A.stride(1) == 1, "A must be contiguous and row major");
    TORCH_CHECK(B.stride(0) == B.size(1) && B.stride(1) == 1, "B must be contiguous and row major");

    int64_t M = A.size(0);
    int64_t N = B.size(0);
    int64_t K = A.size(1);
    auto C = torch::zeros({M, N}, A.options());  // Allocate output tensor

    auto A_ct = torch_to_ct_2d<ct::half_t, ct::GenRowMajor>(A);
    auto B_ct = torch_to_ct_2d<ct::half_t, ct::GenRowMajor>(B);
    auto C_ct = torch_to_ct_2d<ct::half_t, ct::GenRowMajor>(C);

    using KernelTraits = KernelTraits<ct::half_t, 64, 128, 64, 3, true, true>;

    auto [mask, mask_T, mask_table, count] = make_mask<KernelTraits>(M, K, p);
    auto mask_cuda = mask.to(torch::kCUDA, true);
    auto mask_ct = torch_to_ct_2d<ct::uint64_t, ct::GenRowMajor>(mask_cuda);

    matmul<KernelTraits>(A_ct, B_ct, C_ct, mask_ct);

    return {C, mask, mask_T, mask_table, count};
}

// Debugging interface
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t> forward_test(
    torch::Tensor A, torch::Tensor B, torch::Tensor m) {
    TORCH_CHECK(A.dtype() == torch::kHalf, "A must be half");
    TORCH_CHECK(B.dtype() == torch::kHalf, "B must be half");
    TORCH_CHECK(A.type().is_cuda(), "A must be CUDA");
    TORCH_CHECK(B.type().is_cuda(), "B must be CUDA");
    TORCH_CHECK(A.ndimension() == 2, "A must be 2D");
    TORCH_CHECK(B.ndimension() == 2, "B must be 2D");
    TORCH_CHECK(B.size(1) == A.size(1), "B must have the same k dim size as A");
    TORCH_CHECK(A.stride(0) == A.size(1) && A.stride(1) == 1, "A must be contiguous and row major");
    TORCH_CHECK(B.stride(0) == B.size(1) && B.stride(1) == 1, "B must be contiguous and row major");

    int64_t M = A.size(0);
    int64_t N = B.size(0);
    int64_t K = A.size(1);
    auto C = torch::zeros({M, N}, A.options());  // Allocate output tensor

    auto A_ct = torch_to_ct_2d<ct::half_t, ct::GenRowMajor>(A);
    auto B_ct = torch_to_ct_2d<ct::half_t, ct::GenRowMajor>(B);
    auto C_ct = torch_to_ct_2d<ct::half_t, ct::GenRowMajor>(C);

    using KernelTraits = KernelTraits<ct::half_t, 64, 128, 64, 3, true, true>;

    auto [mask, mask_T, mask_table, count] = make_mask_from_existing(m);
    auto mask_cuda = mask.to(torch::kCUDA, true);
    auto mask_ct = torch_to_ct_2d<ct::uint64_t, ct::GenRowMajor>(mask_cuda);

    matmul<KernelTraits>(A_ct, B_ct, C_ct, mask_ct);

    return {C, mask, mask_T, mask_table, count};
}

std::vector<torch::Tensor> backward(torch::Tensor input) { return {input}; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward);
    m.def("forward_test", &forward_test);
    m.def("backward", &backward);
}