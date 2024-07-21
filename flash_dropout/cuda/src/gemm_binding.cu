#include <cute/tensor.hpp>
//
#include <torch/extension.h>

#include "gemm.cuh"
#include "gemm_configs/gemm_config.cuh"

namespace ct = cute;

template <typename T, typename Stride>
auto torch_to_ct_2d(const torch::Tensor &x) {
    auto ptr = ct::make_gmem_ptr(reinterpret_cast<T *>(x.data_ptr()));
    return ct::make_tensor(ptr, ct::make_shape(x.size(0), x.size(1)), Stride{});
}

void check_half_cuda_2d(const torch::Tensor &x) {
    TORCH_CHECK(x.dtype() == torch::kHalf, "X must be half");
    TORCH_CHECK(x.type().is_cuda(), "X must be CUDA");
    TORCH_CHECK(x.ndimension() == 2, "X must be 2D");
}

// C++ interface
torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B) {
    check_half_cuda_2d(A);
    check_half_cuda_2d(B);

    // Allocate output tensor (M x N)
    auto C = torch::empty({A.size(0), B.size(0)}, A.options());

    bool row_major_A = A.stride(1) == 1;
    bool row_major_B = B.stride(1) == 1;
    bool row_major_C = C.stride(1) == 1;
    TORCH_CHECK(row_major_C, "C must be row major");

    auto A_ct_row = torch_to_ct_2d<ct::half_t, ct::GenRowMajor>(A);
    auto A_ct_col = torch_to_ct_2d<ct::half_t, ct::GenColMajor>(A);
    auto B_ct_row = torch_to_ct_2d<ct::half_t, ct::GenRowMajor>(B);
    auto B_ct_col = torch_to_ct_2d<ct::half_t, ct::GenColMajor>(B);
    auto C_ct = torch_to_ct_2d<ct::half_t, ct::GenRowMajor>(C);

    // Dispatch to the correct kernel
    if (row_major_A && row_major_B) {
        gemm<GemmConfigImpl<true, true>>(A_ct_row, B_ct_row, C_ct);
    } else if (row_major_A && !row_major_B) {
        gemm<GemmConfigImpl<true, false>>(A_ct_row, B_ct_col, C_ct);
    } else if (!row_major_A && row_major_B) {
        gemm<GemmConfigImpl<false, true>>(A_ct_col, B_ct_row, C_ct);
    } else {
        gemm<GemmConfigImpl<false, false>>(A_ct_col, B_ct_col, C_ct);
    }
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemm", &gemm_cuda); }