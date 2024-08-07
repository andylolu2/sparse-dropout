#pragma once

#include <cute/tensor.hpp>

#include "gemm.cuh"

namespace ct = cute;

// Define some useful aliases
using ct::_;
using ct::Int;
template <typename T>
using Gmem = ct::ViewEngine<ct::gmem_ptr<T *>>;
template <typename T>
using Smem = ct::ViewEngine<ct::smem_ptr<T *>>;

// Main kernel
template <typename Config, typename LayoutA, typename LayoutB, typename LayoutC, typename LayoutMask>
__global__ void gemm_dsd_kernel(ct::Tensor<Gmem<ct::half_t>, LayoutA> A,
                                ct::Tensor<Gmem<ct::half_t>, LayoutB> B,
                                ct::Tensor<Gmem<ct::half_t>, LayoutC> C,
                                ct::Tensor<Gmem<bool>, LayoutMask> mask,
                                int64_t group_size_m,
                                int64_t block_size,
                                float scale) {
    // Threadblock-level paratitioning
    auto [block_idx_m, block_idx_n] = threadblock_swizzle(blockIdx.x, ct::size<0>(A) / Config::BLK_M, ct::size<0>(B) / Config::BLK_N, group_size_m);
    auto block_shape_A = ct::Shape<Int<Config::BLK_M>, Int<Config::BLK_K>>{};
    auto block_shape_B = ct::Shape<Int<Config::BLK_N>, Int<Config::BLK_K>>{};
    auto block_shape_C = ct::Shape<Int<Config::BLK_M>, Int<Config::BLK_N>>{};
    auto A_blk = ct::local_tile(A, block_shape_A, ct::make_coord(block_idx_m, _));            // BLK_M, BLK_K, N_BLK_K
    auto B_blk = ct::local_tile(B, block_shape_B, ct::make_coord(block_idx_n, _));            // BLK_N, BLK_K, N_BLK_K
    auto C_blk = ct::local_tile(C, block_shape_C, ct::make_coord(block_idx_m, block_idx_n));  // BLK_M, BLK_N

    // Allocate shared memory for the operands
    typename Config::SmemLayoutA smem_layout_A;
    typename Config::SmemLayoutB smem_layout_B;
    __shared__ __align__(sizeof(ct::uint128_t)) ct::half_t sA_data[ct::cosize_v<decltype(smem_layout_A)>];
    __shared__ __align__(sizeof(ct::uint128_t)) ct::half_t sB_data[ct::cosize_v<decltype(smem_layout_B)>];
    auto sA = ct::make_tensor(ct::make_smem_ptr(sA_data), smem_layout_A);
    auto sB = ct::make_tensor(ct::make_smem_ptr(sB_data), smem_layout_B);

    // Main loop
    typename Config::GmemCopyA gmem_copy_A;
    typename Config::GmemCopyB gmem_copy_B;
    SmemGemm<Config, std::decay_t<decltype(C_blk.layout())>> smem_gemm(C_blk);
    for (size_t k = 0; k < ct::size<2>(A_blk); k++) {
        // The corresponding row and col index in the mask
        auto mask_idx_m = block_idx_m * Config::BLK_M / block_size;
        auto mask_idx_k = k * Config::BLK_K / block_size;
        // Check if block is masked
        if (mask(mask_idx_m, mask_idx_k)) {
            continue;
        }
        // Load the k-th A block from gmem to smem
        load_block_from_gmem_to_smem(A_blk(_, _, k), sA, gmem_copy_A);
        // Load the k-th B block from gmem to smem
        load_block_from_gmem_to_smem(B_blk(_, _, k), sB, gmem_copy_B);
        // Wait until all threads have finished loading A and B
        ct::cp_async_wait<0>();
        __syncthreads();
        smem_gemm(sA, sB);
        __syncthreads();
    }
    smem_gemm.scale(scale);
    smem_gemm.write_back();
    ct::cp_async_wait<0>();
}

// Host interface
template <typename Config, typename LayoutA, typename LayoutB, typename LayoutC,
          typename LayoutMask>
void gemm_dsd(const ct::Tensor<Gmem<ct::half_t>, LayoutA> &A,
              const ct::Tensor<Gmem<ct::half_t>, LayoutB> &B,
              const ct::Tensor<Gmem<ct::half_t>, LayoutC> &C,
              const ct::Tensor<Gmem<bool>, LayoutMask> &mask,
              int64_t block_size,
              float scale) {
    assert(ct::size<0>(A) == ct::size<0>(C));  // M
    assert(ct::size<0>(B) == ct::size<1>(C));  // N
    assert(ct::size<1>(A) == ct::size<1>(B));  // K
    int64_t M = ct::size<0>(A);
    int64_t N = ct::size<0>(B);
    int64_t K = ct::size<1>(A);

    // We don't handle predication yet
    assert(M % Config::BLK_M == 0);
    assert(N % Config::BLK_N == 0);
    assert(K % Config::BLK_K == 0);

    // Check mask dims matches A dims
    assert(block_size % Config::BLK_M == 0);
    assert(block_size % Config::BLK_K == 0);
    assert(ct::size<0>(mask) * block_size == M);
    assert(ct::size<1>(mask) * block_size == K);

    dim3 block_dim((M / Config::BLK_M) * (N / Config::BLK_N));
    dim3 thread_dim(Config::NumThreads);

    // We choose group_size_m = sqrt(num_sms) to maximize L2 cache hit rate
    int num_sms;
    CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    int64_t group_size_m = std::sqrt(num_sms);

    gemm_dsd_kernel<Config><<<block_dim, thread_dim>>>(A, B, C, mask, group_size_m, block_size, scale);
}