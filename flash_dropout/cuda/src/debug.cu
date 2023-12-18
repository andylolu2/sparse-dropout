#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>
//
#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/tensor_view_io.h>

#include <bitset>
#include <cute/arch/copy.hpp>
#include <cute/arch/mma_sm75.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>

#include "matmul.cuh"

namespace ct = cute;

// template <int ThreadCount, class TileM, class TileK>
// auto make_gmem_tiled_copy() {
//     using copy_op = ct::AutoVectorizingCopyWithAssumedAlignment<128>;
//     auto copy_atom = ct::Copy_Atom<copy_op, ct::half_t>{};

//     auto tiled_copy = ct::make_tiled_copy(
//         copy_atom,
//         ct::Layout<ct::Shape<TileM, TileK>, ct::Stride<TileK, ct::_1>>{});

//     return tiled_copy;
// }

template <typename scalar_t, typename Layout>
auto host_tensor_to_ct_tensor_row_major(cutlass::HostTensor<scalar_t, Layout>& tensor,
                                        bool transpose = false) {
    auto view_engine = ct::make_gmem_ptr(tensor.device_data());
    int64_t row = tensor.extent().row();
    int64_t col = tensor.extent().column();
    int64_t stride = tensor.stride(0);

    if (std::is_same_v<Layout, cutlass::layout::RowMajor>) {
        if (transpose) {
            throw std::runtime_error("Unsupported transpose");
        } else {
            return ct::make_tensor(view_engine, ct::make_layout(ct::make_shape(row, col),
                                                                ct::make_stride(stride, Int<1>{})));
        }
    } else if (std::is_same_v<Layout, cutlass::layout::ColumnMajor>) {
        if (transpose) {
            return ct::make_tensor(view_engine, ct::make_layout(ct::make_shape(col, row),
                                                                ct::make_stride(stride, Int<1>{})));
        } else {
            throw std::runtime_error("Unsupported transpose");
        }
    } else {
        throw std::runtime_error("Unsupported layout");
    }
}

template <typename scalar_t, typename Layout>
auto host_tensor_to_ct_tensor_col_major(cutlass::HostTensor<scalar_t, Layout>& tensor,
                                        bool transpose = false) {
    auto view_engine = ct::make_gmem_ptr(tensor.device_data());
    int64_t row = tensor.extent().row();
    int64_t col = tensor.extent().column();
    int64_t stride = tensor.stride(0);

    if (std::is_same_v<Layout, cutlass::layout::RowMajor>) {
        if (transpose) {
            return ct::make_tensor(view_engine, ct::make_layout(ct::make_shape(col, row),
                                                                ct::make_stride(Int<1>{}, stride)));
        } else {
            throw std::runtime_error("Unsupported transpose");
        }
    } else if (std::is_same_v<Layout, cutlass::layout::ColumnMajor>) {
        if (transpose) {
            throw std::runtime_error("Unsupported transpose");
        } else {
            return ct::make_tensor(view_engine, ct::make_layout(ct::make_shape(row, col),
                                                                ct::make_stride(Int<1>{}, stride)));
        }
    } else {
        throw std::runtime_error("Unsupported layout");
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " M N K" << std::endl;
        return 1;
    }
    int64_t M = std::atoi(argv[1]);
    int64_t N = std::atoi(argv[2]);
    int64_t K = std::atoi(argv[3]);

    using scalar_t = ct::half_t;
    using layout_A = cutlass::layout::ColumnMajor;
    using layout_B = cutlass::layout::RowMajor;
    using layout_C = cutlass::layout::RowMajor;
    using layout_Mask = cutlass::layout::PackedVectorLayout;
    using Coord = layout_Mask::TensorCoord;
    using KernelTraits =
        KernelTraits<scalar_t, 64, 64, 32, 2, std::is_same_v<layout_A, cutlass::layout::RowMajor>,
                     std::is_same_v<layout_B, cutlass::layout::ColumnMajor>>;

    int64_t BLK_M = KernelTraits::BLK_M;
    int64_t BLK_K = KernelTraits::BLK_K;

    cutlass::HostTensor<scalar_t, layout_A> A({M, K});
    cutlass::HostTensor<scalar_t, layout_B> B({K, N});
    cutlass::HostTensor<scalar_t, layout_C> C({M, N});
    cutlass::HostTensor<scalar_t, layout_C> C_ref({M, N});

    cutlass::reference::host::TensorFillRandomGaussian(A.host_view(), 0);
    cutlass::reference::host::TensorFillRandomGaussian(B.host_view(), 0);
    cutlass::reference::host::TensorFill(C.host_view(), ct::half_t(0));
    cutlass::reference::host::TensorFill(C_ref.host_view(), ct::half_t(0));

    cutlass::DeviceAllocation<ct::uint128_t> mask_data(M / BLK_M);
    auto [mask, mask_T, mask_table] = make_mask(M / BLK_M, K / BLK_K, 0.5);
    mask_data.copy_from_host(mask.data());

    std::cout << "mask: " << std::endl;
    for (int64_t i = 0; i < M / BLK_M; ++i) {
        auto m = mask[i];
        std::cout << std::bitset<64>(m.hilo_.hi) << std::bitset<64>(m.hilo_.lo) << std::endl;
    }
    std::cout << "mask_T: " << std::endl;
    for (int64_t i = 0; i < K / BLK_K; ++i) {
        auto m = mask_T[i];
        std::cout << std::bitset<64>(m.hilo_.hi) << std::bitset<64>(m.hilo_.lo) << std::endl;
    }

    A.sync_device();
    B.sync_device();
    C.sync_device();

    auto A_ct = host_tensor_to_ct_tensor_col_major(A);
    auto B_ct = host_tensor_to_ct_tensor_col_major(B, true);
    auto C_ct = host_tensor_to_ct_tensor_row_major(C);
    auto mask_ct = ct::make_tensor(ct::make_gmem_ptr(mask_data.get()), ct::make_shape(M / BLK_M));

    std::cout << "A layout: " << A_ct.layout() << std::endl;
    std::cout << "B layout: " << B_ct.layout() << std::endl;
    std::cout << "C layout: " << C_ct.layout() << std::endl;
    std::cout << "mask layout: " << mask_ct.layout() << std::endl;

    cutlass::reference::host::Gemm<scalar_t, layout_A, scalar_t, layout_B, scalar_t, layout_C,
                                   ct::half_t, float>
        reference_gemm;
    reference_gemm({int(M), int(N), int(K)}, ct::half_t(1), A.host_ref(), B.host_ref(),
                   ct::half_t(0), C_ref.host_ref());
    // std::cout << "C_ref" << std::endl << C_ref.host_view() << std::endl;

    matmul<KernelTraits>(A_ct, B_ct, C_ct, mask_ct);

    C.sync_host();
    // std::cout << "C" << std::endl << C.host_view() << std::endl;

    float max_rel_err = 0.0f;
    float max_abs_err = 0.0f;

    // Find the max diff
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float c = C.host_ref().at({i, j});
            float c_ref = C_ref.host_ref().at({i, j});
            float diff = std::abs(c - c_ref);
            float rel = diff / std::abs(c_ref);
            max_abs_err = std::max(max_abs_err, diff);
            max_rel_err = std::max(max_rel_err, rel);
        }
    }
    std::cout << "Max abs err: " << max_abs_err << std::endl;
    std::cout << "Max rel err: " << max_rel_err * 100 << "%" << std::endl;

    return 0;
}
