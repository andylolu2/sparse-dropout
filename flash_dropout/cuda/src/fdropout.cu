#include <cute/tensor.hpp>
//
#include <cutlass/util/device_memory.h>
#include <torch/extension.h>

#include <vector>

#include "matmul.cuh"

#ifndef JIT_BLK_M
#define JIT_BLK_M 64
#endif

#ifndef JIT_BLK_K
#define JIT_BLK_K 64
#endif

namespace ct = cute;

template <typename T, typename Layout>
auto torch_to_ct_2d(torch::Tensor x) {
    auto ptr = ct::make_gmem_ptr(reinterpret_cast<T *>(x.data_ptr()));
    return ct::make_tensor(ptr, ct::make_shape(x.size(0), x.size(1)), Layout{});
}

void check_half_cuda_2d_row_major(torch::Tensor x) {
    TORCH_CHECK(x.dtype() == torch::kHalf, "X must be half");
    TORCH_CHECK(x.type().is_cuda(), "X must be CUDA");
    TORCH_CHECK(x.ndimension() == 2, "X must be 2D");
    TORCH_CHECK(x.stride(0) == x.size(1) && x.stride(1) == 1, "X must be contiguous and row major");
}

// C++ interface

// --- forward ---
template <typename KernelTraits>
torch::Tensor forward_core(torch::Tensor A, torch::Tensor B, float p, torch::Tensor mask) {
    check_half_cuda_2d_row_major(A);
    check_half_cuda_2d_row_major(B);
    TORCH_CHECK(B.size(1) == A.size(1), "B must have the same K dim size as A");

    int64_t M = A.size(0);
    int64_t N = B.size(0);
    int64_t K = A.size(1);
    auto C = torch::empty({M, N}, A.options());  // Allocate output tensor

    auto A_ct = torch_to_ct_2d<ct::half_t, ct::GenRowMajor>(A);
    auto B_ct = torch_to_ct_2d<ct::half_t, ct::GenRowMajor>(B);
    auto C_ct = torch_to_ct_2d<ct::half_t, ct::GenRowMajor>(C);

    auto mask_cuda = mask.to(torch::kCUDA, true);
    auto mask_ct = torch_to_ct_2d<ct::uint64_t, ct::GenRowMajor>(mask_cuda);

    matmul<KernelTraits>(A_ct, B_ct, C_ct, mask_ct, static_cast<ct::half_t>(1 / (1 - p)));

    return C;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t> forward(
    torch::Tensor A, torch::Tensor B, float p) {
    using KernelTraits = KernelTraits<ct::half_t, JIT_BLK_M, 128, JIT_BLK_K, 3, true, true, true>;

    auto [mask, mask_T, mask_table, count] = make_mask<KernelTraits>(A.size(0), A.size(1), p);
    auto C = forward_core<KernelTraits>(A, B, p, mask);
    return {C, mask, mask_T, mask_table, count};
}

// Debugging interface
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t> forward_test(
    torch::Tensor A, torch::Tensor B, torch::Tensor m, float p) {
    using KernelTraits = KernelTraits<ct::half_t, JIT_BLK_M, 128, JIT_BLK_K, 3, true, true, true>;

    auto [mask, mask_T, mask_table, count] = make_mask_from_existing(m);
    auto C = forward_core<KernelTraits>(A, B, p, mask);
    return {C, mask, mask_T, mask_table, count};
}

std::vector<torch::Tensor> backward(torch::Tensor dC, torch::Tensor A, torch::Tensor B,
                                    torch::Tensor mask_T, torch::Tensor mask_table, float p,
                                    int64_t count) {
    check_half_cuda_2d_row_major(dC);
    check_half_cuda_2d_row_major(A);
    check_half_cuda_2d_row_major(B);
    TORCH_CHECK(dC.size(0) == A.size(0), "dC must have the same M dim size as A");
    TORCH_CHECK(dC.size(1) == B.size(0), "dC must have the same N dim size as B");

    int64_t M = A.size(0);
    int64_t N = B.size(0);
    int64_t K = A.size(1);
    // auto dA = torch::zeros({M, K}, A.options());
    auto dB = torch::empty({N, K}, B.options());

    auto A_T_ct = torch_to_ct_2d<ct::half_t, ct::GenColMajor>(A.t());
    // auto B_T_ct = torch_to_ct_2d<ct::half_t, ct::GenColMajor>(B.t());
    // auto dA_T_ct = torch_to_ct_2d<ct::half_t, ct::GenRowMajor>(dA);
    auto dB_T_ct = torch_to_ct_2d<ct::half_t, ct::GenColMajor>(dB.t());
    auto dC_T_ct = torch_to_ct_2d<ct::half_t, ct::GenColMajor>(dC.t());

    // TODO: compute dA

    auto mask_T_cuda = mask_T.to(torch::kCUDA, true);
    auto mask_T_ct = torch_to_ct_2d<ct::uint64_t, ct::GenRowMajor>(mask_T_cuda);
    using KernelTraits =
        KernelTraits<ct::half_t, JIT_BLK_K, 128, JIT_BLK_M, 3, false, false, false>;
    matmul<KernelTraits>(A_T_ct, dC_T_ct, dB_T_ct, mask_T_ct, static_cast<ct::half_t>(1 / (1 - p)));

    return {dB};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward);
    m.def("forward_test", &forward_test);
    m.def("backward", &backward);
}