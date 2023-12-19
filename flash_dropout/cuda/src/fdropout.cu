#include <cute/tensor.hpp>
//
#include <torch/extension.h>

#include <vector>

#include "matmul.cuh"

// CUDA forward declarations

namespace ct = cute;

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> forward(torch::Tensor A, torch::Tensor B) {
    using T = ct::half_t;
    int64_t M = A.size(0);
    int64_t N = B.size(0);
    int64_t K = A.size(1);

    auto C = torch::zeros({M, N}, A.options());
    auto A_ct =
        ct::make_tensor(ct::make_gmem_ptr(A.data_ptr<T>()),
                        ct::make_layout(ct::make_shape(M, K), ct::make_stride(K, Int<1>{})));
    // auto B_ct = ct::make_tensor(ct::make_gmem_ptr(B.data_ptr<T>()),
    //                             ct::make_layout(ct::make_shape(N, K), ct::GenRowMajor{}));
    // auto C_ct = ct::make_tensor(ct::make_gmem_ptr(C.data_ptr<T>()),
    //                             ct::make_layout(ct::make_shape(M, N), ct::GenRowMajor{}));

    // using KernelTraits = KernelTraits<T, 32, 32, 64, 2, true, true>;

    // auto [mask_host, mask_T_host, mask_table_host] = make_mask_data<KernelTraits>(M, K, 0.0);
    // cutlass::DeviceAllocation<typename KernelTraits::TMask> mask_data(mask_host.size());
    // cutlass::DeviceAllocation<typename KernelTraits::TMask> mask_T_data(mask_T_host.size());
    // cutlass::DeviceAllocation<int64_t> mask_table_data(mask_table_host.size());
    // mask_data.copy_from_host(mask_host.data());
    // mask_T_data.copy_from_host(mask_T_host.data());
    // mask_table_data.copy_from_host(mask_table_host.data());
    // auto [mask, mask_T, mask_table] = make_mask<KernelTraits>(
    //     mask_data.get(), mask_T_data.get(), mask_table_data.get(), mask_table_data.size(), M, K);

    // matmul<KernelTraits>(A_ct, B_ct, C_ct, mask);

    return {C};
}

std::vector<torch::Tensor> backward(torch::Tensor input) {
    CHECK_INPUT(input);

    return {input};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LLTM forward (CUDA)");
    m.def("backward", &backward, "LLTM backward (CUDA)");
}