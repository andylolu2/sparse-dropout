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

    size_t n_warpups = 100;
    GPU_Clock clock;

    auto data_A = cutlass::DeviceAllocation<ct::half_t>(M * K);
    auto data_B = cutlass::DeviceAllocation<ct::half_t>(N * K);
    auto data_C = cutlass::DeviceAllocation<ct::half_t>(M * N);

    auto A = ct::make_tensor(ct::make_gmem_ptr(data_A.get()),
                             ct::make_layout(ct::make_shape(M, K), ct::make_stride(K, Int<1>{})));
    auto B = ct::make_tensor(ct::make_gmem_ptr(data_B.get()),
                             ct::make_layout(ct::make_shape(N, K), ct::make_stride(K, Int<1>{})));
    auto C = ct::make_tensor(ct::make_gmem_ptr(data_C.get()),
                             ct::make_layout(ct::make_shape(M, N), ct::make_stride(N, Int<1>{})));

    for (size_t i = 0; i < n_warpups; i++) {
        matmul(A, B, C);
    }

    clock.start();
    for (size_t i = 0; i < n_repeats; i++) {
        matmul(A, B, C);
    }
    auto duration = clock.seconds();

    auto flops = 2.0 * M * N * K / duration * n_repeats;
    std::cout << "TFLOPs: " << flops / 1e12 << std::endl;

    return 0;
}
