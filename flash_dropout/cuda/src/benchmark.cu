#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>
//
#include <cute/arch/copy.hpp>
#include <cute/arch/mma_sm75.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/util/GPU_Clock.hpp>

#include "matmul.cuh"

namespace ct = cute;

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " M N K iter" << std::endl;
        return 1;
    }
    int64_t M = std::atoi(argv[1]);
    int64_t N = std::atoi(argv[2]);
    int64_t K = std::atoi(argv[3]);
    size_t n_repeats = std::atoi(argv[4]);

    using scalar_t = ct::half_t;
    using LayoutA = ct::GenRowMajor;
    using LayoutB = ct::GenRowMajor;
    using KernelTraits =
        KernelTraits<scalar_t, 64, 64, 32, 2, std::is_same_v<LayoutA, ct::GenRowMajor>,
                     std::is_same_v<LayoutB, ct::GenRowMajor>>;

    int64_t BLK_M = KernelTraits::BLK_M;
    int64_t BLK_K = KernelTraits::BLK_K;

    size_t n_warpups = 100;
    GPU_Clock clock;

    cutlass::DeviceAllocation<scalar_t> data_A(M * K);
    cutlass::DeviceAllocation<scalar_t> data_B(N * K);
    cutlass::DeviceAllocation<scalar_t> data_C(M * N);
    cutlass::DeviceAllocation<ct::uint128_t> mask_data(M / BLK_M);

    auto A = ct::make_tensor(ct::make_gmem_ptr(data_A.get()),
                             ct::make_layout(ct::make_shape(M, K), LayoutA{}));
    auto B = ct::make_tensor(ct::make_gmem_ptr(data_B.get()),
                             ct::make_layout(ct::make_shape(N, K), LayoutB{}));
    auto C = ct::make_tensor(ct::make_gmem_ptr(data_C.get()),
                             ct::make_layout(ct::make_shape(M, N), ct::GenRowMajor{}));
    auto mask = ct::make_tensor(ct::make_gmem_ptr(mask_data.get()), ct::make_shape(M / BLK_M));

    for (size_t i = 0; i < n_warpups; i++) {
        auto [mask_host, mask_T, mask_table] = make_mask(M / BLK_M, K / BLK_K, 0.0);
        // std::cout << "Mask proportion: " << double(mask_table.size()) / (M / BLK_M * K / BLK_K)
        //           << std::endl;
        mask_data.copy_from_host(mask_host.data());
        matmul<KernelTraits>(A, B, C, mask);
    }

    clock.start();
    for (size_t i = 0; i < n_repeats; i++) {
        auto [mask_host, mask_T, mask_table] = make_mask(M / BLK_M, K / BLK_K, 0.0);
        mask_data.copy_from_host(mask_host.data());
        matmul<KernelTraits>(A, B, C, mask);
    }
    auto duration = clock.seconds();

    auto flops = 2.0 * M * N * K / duration * n_repeats;
    std::cout << "TFLOPs: " << flops / 1e12 << std::endl;

    return 0;
}
