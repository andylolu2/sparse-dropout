#include <torch/extension.h>

#include <cute/tensor.hpp>

#include "gemm.cuh"
#include "gemm_configs/gemm_config.cuh"
#include "matmul_dsd.cuh"
#include "matmul_sdd.cuh"

namespace ct = cute;

template <typename T, typename Layout>
auto torch_to_ct_2d(const torch::Tensor &x) {
    auto ptr = ct::make_gmem_ptr(reinterpret_cast<T *>(x.data_ptr()));
    return ct::make_tensor(ptr, ct::make_shape(x.size(0), x.size(1)), Layout{});
}

void check_operand(const torch::Tensor &x) {
    TORCH_CHECK(x.type().is_cuda(), "X must be CUDA");
    TORCH_CHECK(x.stride(0) == 1 || x.stride(1) == 1, "X must be row or column major");
    TORCH_CHECK(x.ndimension() == 2, "X must be 2D");
}

template <bool RowMajorA, bool RowMajorB>
torch::Tensor gemm_cuda_inner(torch::Tensor A, torch::Tensor B) {
    // Allocate output tensor (M x N)
    auto C = torch::empty({A.size(0), B.size(0)}, A.options());

    auto A_ct = torch_to_ct_2d<ct::half_t, std::conditional_t<RowMajorA, ct::GenRowMajor, ct::GenColMajor>>(A);
    auto B_ct = torch_to_ct_2d<ct::half_t, std::conditional_t<RowMajorB, ct::GenRowMajor, ct::GenColMajor>>(B);
    auto C_ct = torch_to_ct_2d<ct::half_t, ct::GenRowMajor>(C);

    gemm<GemmConfigImpl<RowMajorA, RowMajorB>>(A_ct, B_ct, C_ct);
    return C;
}

template <bool... RowMajors, typename... Ts>
torch::Tensor gemm_cuda_inner(torch::Tensor A, torch::Tensor B, bool row_major, Ts... row_majors) {
    if (row_major) {
        return gemm_cuda_inner<RowMajors..., true>(A, B, row_majors...);
    } else {
        return gemm_cuda_inner<RowMajors..., false>(A, B, row_majors...);
    }
}

torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B) {
    check_operand(A);
    check_operand(B);
    TORCH_CHECK(A.dtype() == torch::kHalf, "A must be half");
    TORCH_CHECK(B.dtype() == torch::kHalf, "B must be half");
    bool row_major_A = A.stride(1) == 1;
    bool row_major_B = B.stride(1) == 1;
    return gemm_cuda_inner<>(A, B, row_major_A, row_major_B);
}

template <bool RowMajorA, bool RowMajorB, bool RowMajorMask>
torch::Tensor gemm_dsd_cuda_inner(torch::Tensor A, torch::Tensor B, torch::Tensor mask,
                                  int64_t block_size, float scale) {
    // Allocate output tensor (M x N)
    auto C = torch::empty({A.size(0), B.size(0)}, A.options());

    auto A_ct = torch_to_ct_2d<ct::half_t, std::conditional_t<RowMajorA, ct::GenRowMajor, ct::GenColMajor>>(A);
    auto B_ct = torch_to_ct_2d<ct::half_t, std::conditional_t<RowMajorB, ct::GenRowMajor, ct::GenColMajor>>(B);
    auto C_ct = torch_to_ct_2d<ct::half_t, ct::GenRowMajor>(C);
    auto mask_ct = torch_to_ct_2d<bool, std::conditional_t<RowMajorMask, ct::GenRowMajor, ct::GenColMajor>>(mask);

    gemm_dsd<GemmConfigImpl<RowMajorA, RowMajorB>>(A_ct, B_ct, C_ct, mask_ct, block_size, scale);
    return C;
}

template <bool... RowMajors, typename... Ts>
torch::Tensor gemm_dsd_cuda_inner(torch::Tensor A, torch::Tensor B, torch::Tensor mask,
                                  int64_t block_size, float scale, bool row_major,
                                  Ts... row_majors) {
    if (row_major) {
        return gemm_dsd_cuda_inner<RowMajors..., true>(A, B, mask, block_size, scale, row_majors...);
    } else {
        return gemm_dsd_cuda_inner<RowMajors..., false>(A, B, mask, block_size, scale, row_majors...);
    }
}

torch::Tensor gemm_dsd_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor mask,
                            int64_t block_size, float scale) {
    check_operand(A);
    check_operand(B);
    check_operand(mask);
    TORCH_CHECK(A.dtype() == torch::kHalf, "A must be half");
    TORCH_CHECK(B.dtype() == torch::kHalf, "B must be half");
    TORCH_CHECK(mask.dtype() == torch::kBool, "Mask must be bool");
    bool row_major_A = A.stride(1) == 1;
    bool row_major_B = B.stride(1) == 1;
    bool row_major_mask = mask.stride(1) == 1;
    return gemm_dsd_cuda_inner<>(A, B, mask, block_size, scale, row_major_A, row_major_B, row_major_mask);
}

template <bool RowMajorA, bool RowMajorB, bool RowMajorMask>
torch::Tensor gemm_sdd_cuda_inner(torch::Tensor A, torch::Tensor B, torch::Tensor mask,
                                  int64_t block_size, float scale) {
    // Allocate output tensor (M x N)
    auto C = torch::empty({A.size(0), B.size(0)}, A.options());

    auto A_ct = torch_to_ct_2d<ct::half_t, std::conditional_t<RowMajorA, ct::GenRowMajor, ct::GenColMajor>>(A);
    auto B_ct = torch_to_ct_2d<ct::half_t, std::conditional_t<RowMajorB, ct::GenRowMajor, ct::GenColMajor>>(B);
    auto C_ct = torch_to_ct_2d<ct::half_t, ct::GenRowMajor>(C);
    auto mask_ct = torch_to_ct_2d<bool, std::conditional_t<RowMajorMask, ct::GenRowMajor, ct::GenColMajor>>(mask);

    gemm_sdd<GemmConfigImpl<RowMajorA, RowMajorB>>(A_ct, B_ct, C_ct, mask_ct, block_size, scale);
    return C;
}

template <bool... RowMajors, typename... Ts>
torch::Tensor gemm_sdd_cuda_inner(torch::Tensor A, torch::Tensor B, torch::Tensor mask,
                                  int64_t block_size, float scale, bool row_major,
                                  Ts... row_majors) {
    if (row_major) {
        return gemm_sdd_cuda_inner<RowMajors..., true>(A, B, mask, block_size, scale, row_majors...);
    } else {
        return gemm_sdd_cuda_inner<RowMajors..., false>(A, B, mask, block_size, scale, row_majors...);
    }
}

torch::Tensor gemm_sdd_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor mask,
                            int64_t block_size, float scale) {
    check_operand(A);
    check_operand(B);
    check_operand(mask);
    TORCH_CHECK(A.dtype() == torch::kHalf, "A must be half");
    TORCH_CHECK(B.dtype() == torch::kHalf, "B must be half");
    TORCH_CHECK(mask.dtype() == torch::kBool, "Mask must be bool");
    bool row_major_A = A.stride(1) == 1;
    bool row_major_B = B.stride(1) == 1;
    bool row_major_mask = mask.stride(1) == 1;
    return gemm_sdd_cuda_inner<>(A, B, mask, block_size, scale, row_major_A, row_major_B,
                                 row_major_mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm_cuda);
    m.def("gemm_dsd", &gemm_dsd_cuda);
    m.def("gemm_sdd", &gemm_sdd_cuda);
}