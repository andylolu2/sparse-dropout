#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>
//
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

int main(int argc, char *argv[]) {
    // using MmaAtom = ct::MMA_Atom<ct::SM75_16x8x8_F32F16F16F32_TN>;
    // using TiledMma = ct::TiledMMA<MmaAtom>;

    // auto tiled_mma = ct::make_tiled_mma(ct::SM75_16x8x8_F32F16F16F32_TN{});

    // using copy_op = ct::AutoVectorizingCopyWithAssumedAlignment<128>;
    // auto copy_atom = ct::Copy_Atom<ct::DefaultCopy, ct::half_t>{};

    // auto tiled_copy = ct::make_tiled_copy(
    //     copy_atom,
    //     ct::Layout<ct::Shape<ct::_32, ct::_4>, ct::Stride<ct::_4,
    //     ct::_1>>{});

    // ct::print(tiled_mma);
    // ct::print(tiled_copy);

    // auto data = cutlass::DeviceAllocation<ct::half_t>(128);
    // auto gA =
    //     ct::make_tensor(ct::make_gmem_ptr(data.get()), ct::make_shape(513,
    //     80));

    // ct::print(thr_copy.partition_D(gA));
    int64_t M = 64;
    int64_t N = 64;
    int64_t K = 32;

    auto host_data_A = std::vector<ct::half_t>(M * K);
    for (int i = 0; i < M * K; ++i) {
        host_data_A[i] = ct::half_t(float(i) / 100.0f);
    }
    auto host_data_B = std::vector<ct::half_t>(N * K);
    for (int i = 0; i < N * K; ++i) {
        host_data_B[i] = ct::half_t(-float(i) / 100.0f);
    }
    auto host_data_C = std::vector<ct::half_t>(M * N);
    for (int i = 0; i < M * N; ++i) {
        host_data_C[i] = ct::half_t(0);
    }

    auto data_A = cutlass::DeviceAllocation<ct::half_t>(M * K);
    auto data_B = cutlass::DeviceAllocation<ct::half_t>(N * K);
    auto data_C = cutlass::DeviceAllocation<ct::half_t>(M * N);
    data_A.copy_from_host(host_data_A.data());
    data_B.copy_from_host(host_data_B.data());
    data_C.copy_from_host(host_data_C.data());

    auto A = ct::make_tensor(ct::make_gmem_ptr(data_A.get()),
                             ct::make_layout(ct::make_shape(M, K), ct::make_stride(K, 1L)));
    auto B = ct::make_tensor(ct::make_gmem_ptr(data_B.get()),
                             ct::make_layout(ct::make_shape(N, K), ct::make_stride(K, 1L)));
    auto C = ct::make_tensor(ct::make_gmem_ptr(data_C.get()),
                             ct::make_layout(ct::make_shape(M, N), ct::make_stride(N, 1L)));

    auto host_C = ct::make_tensor(host_data_C.data(), C.layout());
    std::cout << host_C << std::endl;

    matmul<ct::half_t><<<1, 128>>>(A, B, C);

    data_C.copy_to_host(host_data_C.data());
    std::cout << host_C << std::endl;

    // auto shape = ct::make_shape(ct::_8{}, ct::_8{});
    // auto tiled_shape = ct::tile_to_shape(ct::make_layout(shape),
    //                                      ct::make_shape(ct::_32{},
    //                                      ct::_64{}));

    // std::cout << "shape: " << shape << std::endl;
    // std::cout << "tiled_shape: " << tiled_shape << std::endl;

    return 0;
}
