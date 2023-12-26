#pragma once

#include <cute/tensor.hpp>
//
// #include <cutlass/util/device_dump.h>

#include <cute/arch/copy.hpp>
#include <cute/arch/mma_sm75.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <tuple>
#include <vector>

#include "kernel_traits.cuh"

namespace ct = cute;

using ct::_;
using ct::Int;

template <typename scalar_t>
using Gmem = ct::ViewEngine<ct::gmem_ptr<scalar_t*>>;

template <typename scalar_t>
using Smem = ct::ViewEngine<ct::smem_ptr<scalar_t*>>;

template <typename KernelTraits, typename LayoutGaSrc, typename LayoutGaDst, typename LayoutGbSrc,
          typename LayoutGbDst, typename LayoutSa, typename LayoutSb, typename LayoutC,
          typename LayoutMask>
__device__ void matmul_dsd_thread(
    ct::Tensor<Gmem<typename KernelTraits::T>, LayoutGaSrc> gA_to_sA_src,
    ct::Tensor<Smem<typename KernelTraits::T>, LayoutGaDst> gA_to_sA_dst,
    ct::Tensor<Gmem<typename KernelTraits::T>, LayoutGbSrc> gB_to_sB_src,
    ct::Tensor<Smem<typename KernelTraits::T>, LayoutGbDst> gB_to_sB_dst,
    ct::Tensor<Smem<typename KernelTraits::T>, LayoutSa> sA,
    ct::Tensor<Smem<typename KernelTraits::T>, LayoutSb> sB,
    ct::Tensor<Gmem<typename KernelTraits::T>, LayoutC> C_blk,
    ct::Tensor<Gmem<typename KernelTraits::TMask>, LayoutMask> mask_bits,
    typename KernelTraits::T scale) {
    typename KernelTraits::GmemTiledCopyA gmem_tiled_copy_A;
    typename KernelTraits::GmemTiledCopyB gmem_tiled_copy_B;
    typename KernelTraits::GmemTiledCopyC gmem_tiled_copy_C;

    typename KernelTraits::TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    // rA and rB should follow the canonical layout (row major). The copy atom should handle
    // tranposing the layout (if necessary).
    typename KernelTraits::SmemLayoutACanonical smem_layout_A_canonical;
    typename KernelTraits::SmemLayoutBCanonical smem_layout_B_canonical;
    auto sA_can = ct::make_tensor(ct::make_smem_ptr(sA.data()), smem_layout_A_canonical);
    auto sB_can = ct::make_tensor(ct::make_smem_ptr(sB.data()), smem_layout_B_canonical);
    auto rA = thr_mma.partition_fragment_A(sA_can);  // MMA, MMA_M, MMA_K
    auto rB = thr_mma.partition_fragment_B(sB_can);  // MMA, MMA_N, MMA_K
    auto rC = thr_mma.partition_fragment_C(C_blk);   // MMA, MMA_M, MMA_N
    auto gC = thr_mma.partition_C(C_blk);            // Corresponding fragment in gmem to write back
    ct::clear(rC);

    typename KernelTraits::SmemTiledCopyA smem_tiled_copy_A;
    auto thr_copy_A = smem_tiled_copy_A.get_thread_slice(threadIdx.x);
    auto sA_to_rA_src = thr_copy_A.partition_S(sA);  // COPY_V, COPY_M, COPY_K
    auto sA_to_rA_dst = thr_copy_A.retile_D(rA);     // COPY_V, COPY_M, COPY_K

    typename KernelTraits::SmemTiledCopyB smem_tiled_copy_B;
    auto thr_copy_B = smem_tiled_copy_B.get_thread_slice(threadIdx.x);
    auto sB_to_rB_src = thr_copy_B.partition_S(sB);  // COPY_V, COPY_N, COPY_K
    auto sB_to_rB_dst = thr_copy_B.retile_D(rB);     // COPY_V, COPY_N, COPY_K

#if 0
    if (ct::thread0()) {
        ct::print(tiled_mma);
        ct::print(" <- tiled_mma\n");
        ct::print(C_blk);
        ct::print(" <- C_blk\n");
        ct::print(rA);
        ct::print(" <- rA\n");
        ct::print(rB);
        ct::print(" <- rB\n");
        ct::print(rC);
        ct::print(" <- rC\n");
        ct::print(gC);
        ct::print(" <- gC\n");
        ct::print(smem_tiled_copy_A);
        ct::print(" <- smem_tiled_copy_A\n");
        ct::print(sA);
        ct::print(" <- sA\n");
        ct::print(sA_to_rA_src);
        ct::print(" <- sA_to_rA_src\n");
        ct::print(sA_to_rA_dst);
        ct::print(" <- sA_to_rA_dst\n");
        ct::print(smem_tiled_copy_B);
        ct::print(" <- smem_tiled_copy_B\n");
        ct::print(sB);
        ct::print(" <- sB\n");
        ct::print(sB_to_rB_src);
        ct::print(" <- sB_to_rB_src\n");
        ct::print(sB_to_rB_dst);
        ct::print(" <- sB_to_rB_dst\n");
    }
#endif

    int N_BLK_K = ct::size<3>(gA_to_sA_src);
    using TMask = typename KernelTraits::TMask;
    TMask mask_bit_pack;

#pragma unroll
    for (size_t k_blk = 0; k_blk < N_BLK_K; k_blk++) {
        if (k_blk % ct::sizeof_bits_v<TMask> == 0) {
            mask_bit_pack = mask_bits(k_blk / ct::sizeof_bits_v<TMask>);
        }
        bool drop = (mask_bit_pack & 1) == 1;
        mask_bit_pack = mask_bit_pack >> 1;
        if (drop) {
            continue;
        }

        ct::copy(gmem_tiled_copy_A, gA_to_sA_src(_, _, _, k_blk), gA_to_sA_dst);
        ct::copy(gmem_tiled_copy_B, gB_to_sB_src(_, _, _, k_blk), gB_to_sB_dst);
        __syncthreads();

        // Load rA and rB (distributed across registers of threads) by copying from smem to gmem
        ct::copy(smem_tiled_copy_A, sA_to_rA_src, sA_to_rA_dst);
        ct::copy(smem_tiled_copy_B, sB_to_rB_src, sB_to_rB_dst);
#if 0
        cutlass::debug::dump_shmem(sA.data().get(), sA.size());
        if (ct::thread0()) {
            for (size_t i = 0; i < ct::size(rA); i++) {
                printf("%.0f ", static_cast<float>(rA(i)));
            }
            printf("\n");
        }
#endif
        // Perform gemm
        ct::gemm(tiled_mma, rA, rB, rC);
#if 0
        if (ct::thread0()) {
            for (size_t i = 0; i < ct::size(rC); i++) {
                printf("%.4f ", static_cast<float>(rC(i)));
            }
            printf("\n");
        }
#endif
    }

    // Prologue
#pragma unroll
    for (size_t i = 0; i < ct::size(rC); ++i) {
        rC(i) = rC(i) * scale;
    }

    // Write back result
    ct::copy(gmem_tiled_copy_C, rC, gC);
}

template <typename KernelTraits, typename LayoutA, typename LayoutB, typename LayoutC,
          typename LayoutMask>
__device__ void matmul_dsd_threadblock(
    ct::Tensor<Gmem<typename KernelTraits::T>, LayoutA> A_blk,
    ct::Tensor<Gmem<typename KernelTraits::T>, LayoutB> B_blk,
    ct::Tensor<Gmem<typename KernelTraits::T>, LayoutC> C_blk,
    ct::Tensor<Gmem<typename KernelTraits::TMask>, LayoutMask> mask_bits,
    typename KernelTraits::T scale) {
    typename KernelTraits::SmemLayoutA smem_layout_A;
    typename KernelTraits::SmemLayoutB smem_layout_B;

    __shared__ typename KernelTraits::T sA_data[ct::cosize_v<decltype(smem_layout_A)>];
    __shared__ typename KernelTraits::T sB_data[ct::cosize_v<decltype(smem_layout_B)>];
    auto sA = ct::make_tensor(ct::make_smem_ptr(sA_data), smem_layout_A);
    auto sB = ct::make_tensor(ct::make_smem_ptr(sB_data), smem_layout_B);

#if 0
    if (ct::thread0()) {
        ct::print(sA);
        ct::print(" <- sA\n");
        ct::print(sB);
        ct::print(" <- sB\n");
        ct::print(A_blk);
        ct::print(" <- A_blk\n");
        ct::print(B_blk);
        ct::print(" <- B_blk\n");
        ct::print(C_blk);
        ct::print(" <- C_blk\n");
    }
#endif

    typename KernelTraits::GmemTiledCopyA gmem_tiled_copy_A;
    typename KernelTraits::GmemTiledCopyA gmem_tiled_copy_B;
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_thread_slice(threadIdx.x);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_thread_slice(threadIdx.x);

    // Fragments for gmem -> smem copy
    auto gA_to_sA_src = gmem_thr_copy_A.partition_S(A_blk);  // COPY_V, COPY_M, COPY_K, N_BLK_K
    auto gA_to_sA_dst = gmem_thr_copy_A.partition_D(sA);     // COPY_V, COPY_M, COPY_K
    auto gB_to_sB_src = gmem_thr_copy_B.partition_S(B_blk);  // COPY_V, COPY_N, COPY_K, N_BLK_K
    auto gB_to_sB_dst = gmem_thr_copy_B.partition_D(sB);     // COPY_V, COPY_N, COPY_K

#if 0
    if (ct::thread0()) {
        ct::print(gmem_tiled_copy);
        ct::print(" <- gmem_tiled_copy\n");
        ct::print(A_blk);
        ct::print(" <- A_blk\n");
        ct::print(sA);
        ct::print(" <- sA\n");
        ct::print(gA_to_sA_src);
        ct::print(" <- gA_to_sA_src\n");
        ct::print(gA_to_sA_dst);
        ct::print(" <- gA_to_sA_dst\n");
        ct::print(B_blk);
        ct::print(" <- B_blk\n");
        ct::print(sB);
        ct::print(" <- sB\n");
        ct::print(gB_to_sB_src);
        ct::print(" <- gB_to_sB_src\n");
        ct::print(gB_to_sB_dst);
        ct::print(" <- gB_to_sB_dst\n");
    }
#endif

    matmul_dsd_thread<KernelTraits>(gA_to_sA_src, gA_to_sA_dst, gB_to_sB_src, gB_to_sB_dst, sA, sB,
                                    C_blk, mask_bits, scale);
}

template <typename KernelTraits>
__global__ void matmul_dsd_kernel(
    ct::Tensor<Gmem<typename KernelTraits::T>, typename KernelTraits::LayoutA> A,
    ct::Tensor<Gmem<typename KernelTraits::T>, typename KernelTraits::LayoutB> B,
    ct::Tensor<Gmem<typename KernelTraits::T>, typename KernelTraits::LayoutC> C,
    ct::Tensor<Gmem<typename KernelTraits::TMask>, typename KernelTraits::LayoutMask> mask,
    typename KernelTraits::T scale) {
    using BlockShapeA = typename KernelTraits::BlockShapeA;
    using BlockShapeB = typename KernelTraits::BlockShapeB;
    using BlockShapeC = typename KernelTraits::BlockShapeC;

    auto A_blk_all = ct::tiled_divide(A, BlockShapeA{});  // (BLK_M, BLK_K), N_BLK_M, N_BLK_K
    auto B_blk_all = ct::tiled_divide(B, BlockShapeB{});  // (BLK_N, BLK_K), N_BLK_N, N_BLK_K
    auto C_blk_all = ct::tiled_divide(C, BlockShapeC{});  // (BLK_M, BLK_N), N_BLK_M, N_BLK_N

    // Threadblock swizzling
    int N_BLK_M = ct::size<1>(A_blk_all);
    int N_BLK_N = ct::size<1>(B_blk_all);
    int blocks_per_group = KernelTraits::GroupSizeM * N_BLK_N;
    int first_block_idx_m = (blockIdx.x / blocks_per_group) * KernelTraits::GroupSizeM;
    int group_size_m = min(N_BLK_M - first_block_idx_m, KernelTraits::GroupSizeM);  // Edge case
    int block_idx_m = first_block_idx_m + (blockIdx.x % group_size_m);
    int block_idx_n = (blockIdx.x % blocks_per_group) / group_size_m;

    auto A_blk = ct::flatten(A_blk_all(_, block_idx_m, _));            // BLK_M, BLK_K, N_BLK_K
    auto B_blk = ct::flatten(B_blk_all(_, block_idx_n, _));            // BLK_N, BLK_K, N_BLK_K
    auto C_blk = ct::flatten(C_blk_all(_, block_idx_m, block_idx_n));  // BLK_M, BLK_N

    auto mask_bits = mask(block_idx_m, _);

#if 0
    if (ct::thread0()) {
        ct::print(A);
        ct::print(" <- A\n");
        ct::print(B);
        ct::print(" <- B\n");
        ct::print(C);
        ct::print(" <- C\n");
        ct::print(A_blk_all);
        ct::print(" <- A_blk_all\n");
        ct::print(B_blk_all);
        ct::print(" <- B_blk_all\n");
        ct::print(C_blk_all);
        ct::print(" <- C_blk_all\n");
        ct::print(A_blk);
        ct::print(" <- A_blk\n");
        ct::print(B_blk);
        ct::print(" <- B_blk\n");
        ct::print(C_blk);
        ct::print(" <- C_blk\n");
        ct::print(mask);
        ct::print(" <- mask\n");
    }
#endif

    matmul_dsd_threadblock<KernelTraits>(A_blk, B_blk, C_blk, mask_bits, scale);
}

template <typename KernelTraits>
void matmul_dsd(
    ct::Tensor<Gmem<typename KernelTraits::T>, typename KernelTraits::LayoutA> A,
    ct::Tensor<Gmem<typename KernelTraits::T>, typename KernelTraits::LayoutB> B,
    ct::Tensor<Gmem<typename KernelTraits::T>, typename KernelTraits::LayoutC> C,
    ct::Tensor<Gmem<typename KernelTraits::TMask>, typename KernelTraits::LayoutMask> mask,
    typename KernelTraits::T scale) {
    assert(ct::size<0>(A) == ct::size<0>(C));  // M
    assert(ct::size<0>(B) == ct::size<1>(C));  // N
    assert(ct::size<1>(A) == ct::size<1>(B));  // K

    size_t M = ct::size<0>(A);
    size_t N = ct::size<0>(B);
    size_t K = ct::size<1>(A);
    size_t BLK_M = KernelTraits::BLK_M;
    size_t BLK_N = KernelTraits::BLK_N;
    size_t BLK_K = KernelTraits::BLK_K;

    // We don't handle predication yet
    assert(M % BLK_M == 0);
    assert(N % BLK_N == 0);
    assert(K % BLK_K == 0);

    size_t N_BLK_M = M / BLK_M;
    size_t N_BLK_N = N / BLK_N;
    size_t N_BLK_K = K / BLK_K;

    assert(ct::size<0>(mask) == N_BLK_M);
    assert(ct::size<1>(mask) * ct::sizeof_bits_v<typename KernelTraits::TMask> >= N_BLK_K);

    dim3 block_dim(N_BLK_M * N_BLK_N);
    dim3 thread_dim(KernelTraits::NumThreads);

    matmul_dsd_kernel<KernelTraits><<<block_dim, thread_dim>>>(A, B, C, mask, scale);
}