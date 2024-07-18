#pragma once

#include <cute/tensor.hpp>
//
#include <torch/extension.h>
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
          typename LayoutGbDst, typename LayoutSa, typename LayoutSb, typename LayoutC>
__device__ void matmul_sdd_thread(
    ct::Tensor<Gmem<typename KernelTraits::T>, LayoutGaSrc> gA_to_sA_src,
    ct::Tensor<Smem<typename KernelTraits::T>, LayoutGaDst> gA_to_sA_dst,
    ct::Tensor<Gmem<typename KernelTraits::T>, LayoutGbSrc> gB_to_sB_src,
    ct::Tensor<Smem<typename KernelTraits::T>, LayoutGbDst> gB_to_sB_dst,
    ct::Tensor<Smem<typename KernelTraits::T>, LayoutSa> sA,
    ct::Tensor<Smem<typename KernelTraits::T>, LayoutSb> sB,
    ct::Tensor<Gmem<typename KernelTraits::T>, LayoutC> C_blk, typename KernelTraits::T scale) {
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

    int N_BLK_K = ct::size<3>(gA_to_sA_src);

#pragma unroll
    for (size_t k_blk = 0; k_blk < N_BLK_K; k_blk++) {
        ct::copy(gmem_tiled_copy_A, gA_to_sA_src(_, _, _, k_blk), gA_to_sA_dst);
        ct::copy(gmem_tiled_copy_B, gB_to_sB_src(_, _, _, k_blk), gB_to_sB_dst);
        __syncthreads();

        // Load rA and rB (distributed across registers of threads) by copying from smem to gmem
        ct::copy(smem_tiled_copy_A, sA_to_rA_src, sA_to_rA_dst);
        ct::copy(smem_tiled_copy_B, sB_to_rB_src, sB_to_rB_dst);
        // Perform gemm
        ct::gemm(tiled_mma, rA, rB, rC);
    }

    // Prologue
#pragma unroll
    for (size_t i = 0; i < ct::size(rC); ++i) {
        rC(i) = rC(i) * scale;
    }

    // Write back result
    ct::copy(gmem_tiled_copy_C, rC, gC);
}

template <typename KernelTraits, typename LayoutA, typename LayoutB, typename LayoutC>
__device__ void matmul_sdd_threadblock(ct::Tensor<Gmem<typename KernelTraits::T>, LayoutA> A_blk,
                                       ct::Tensor<Gmem<typename KernelTraits::T>, LayoutB> B_blk,
                                       ct::Tensor<Gmem<typename KernelTraits::T>, LayoutC> C_blk,
                                       typename KernelTraits::T scale) {
    typename KernelTraits::SmemLayoutA smem_layout_A;
    typename KernelTraits::SmemLayoutB smem_layout_B;

    __shared__ typename KernelTraits::T sA_data[ct::cosize_v<decltype(smem_layout_A)>];
    __shared__ typename KernelTraits::T sB_data[ct::cosize_v<decltype(smem_layout_B)>];
    auto sA = ct::make_tensor(ct::make_smem_ptr(sA_data), smem_layout_A);
    auto sB = ct::make_tensor(ct::make_smem_ptr(sB_data), smem_layout_B);

    typename KernelTraits::GmemTiledCopyA gmem_tiled_copy_A;
    typename KernelTraits::GmemTiledCopyA gmem_tiled_copy_B;
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_thread_slice(threadIdx.x);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_thread_slice(threadIdx.x);

    // Fragments for gmem -> smem copy
    auto gA_to_sA_src = gmem_thr_copy_A.partition_S(A_blk);  // COPY_V, COPY_M, COPY_K, N_BLK_K
    auto gA_to_sA_dst = gmem_thr_copy_A.partition_D(sA);     // COPY_V, COPY_M, COPY_K
    auto gB_to_sB_src = gmem_thr_copy_B.partition_S(B_blk);  // COPY_V, COPY_N, COPY_K, N_BLK_K
    auto gB_to_sB_dst = gmem_thr_copy_B.partition_D(sB);     // COPY_V, COPY_N, COPY_K

    matmul_sdd_thread<KernelTraits>(gA_to_sA_src, gA_to_sA_dst, gB_to_sB_src, gB_to_sB_dst, sA, sB,
                                    C_blk, scale);
}

template <typename KernelTraits>
__global__ void matmul_sdd_kernel(
    ct::Tensor<Gmem<typename KernelTraits::T>, typename KernelTraits::LayoutA> A,
    ct::Tensor<Gmem<typename KernelTraits::T>, typename KernelTraits::LayoutB> B,
    ct::Tensor<Gmem<typename KernelTraits::T>, typename KernelTraits::LayoutC> C,
    ct::Tensor<Gmem<ct::int64_t>, typename KernelTraits::LayoutMaskTable> mask_table,
    typename KernelTraits::T scale) {
    using BlockShapeA = typename KernelTraits::BlockShapeA;
    using BlockShapeB = typename KernelTraits::BlockShapeB;
    using BlockShapeC = typename KernelTraits::BlockShapeC;

    auto A_blk_all = ct::tiled_divide(A, BlockShapeA{});  // (BLK_M, BLK_K), N_BLK_M, N_BLK_K
    auto B_blk_all = ct::tiled_divide(B, BlockShapeB{});  // (BLK_N, BLK_K), N_BLK_N, N_BLK_K
    auto C_blk_all = ct::tiled_divide(C, BlockShapeC{});  // (BLK_M, BLK_N), N_BLK_M, N_BLK_N

    auto block_indices = ct::make_tensor<ct::int64_t>(ct::make_shape(Int<2>{}));
    ct::Copy_Atom<ct::AutoVectorizingCopyWithAssumedAlignment<128>, ct::int64_t> copy_atom;
    ct::copy(copy_atom, mask_table(blockIdx.x, _), block_indices);
    int64_t block_idx_m = block_indices(0);
    int64_t block_idx_n = block_indices(1);

    auto A_blk = ct::flatten(A_blk_all(_, block_idx_m, _));            // BLK_M, BLK_K, N_BLK_K
    auto B_blk = ct::flatten(B_blk_all(_, block_idx_n, _));            // BLK_N, BLK_K, N_BLK_K
    auto C_blk = ct::flatten(C_blk_all(_, block_idx_m, block_idx_n));  // BLK_M, BLK_N

    matmul_sdd_threadblock<KernelTraits>(A_blk, B_blk, C_blk, scale);
}

template <typename KernelTraits>
void matmul_sdd(ct::Tensor<Gmem<typename KernelTraits::T>, typename KernelTraits::LayoutA> A,
                ct::Tensor<Gmem<typename KernelTraits::T>, typename KernelTraits::LayoutB> B,
                ct::Tensor<Gmem<typename KernelTraits::T>, typename KernelTraits::LayoutC> C,
                ct::Tensor<Gmem<ct::int64_t>, typename KernelTraits::LayoutMaskTable> mask_table,
                int64_t count, typename KernelTraits::T scale) {
    assert(ct::size<0>(A) == ct::size<0>(C));  // M
    assert(ct::size<0>(B) == ct::size<1>(C));  // N
    assert(ct::size<1>(A) == ct::size<1>(B));  // K

    // We don't handle predication yet
    assert(ct::size<0>(A) % KernelTraits::BLK_M == 0);
    assert(ct::size<0>(B) % KernelTraits::BLK_N == 0);
    assert(ct::size<1>(A) % KernelTraits::BLK_K == 0);

    if (count == 0) {
        return;
    }

    dim3 block_dim(count);
    dim3 thread_dim(KernelTraits::NumThreads);

    matmul_sdd_kernel<KernelTraits><<<block_dim, thread_dim>>>(A, B, C, mask_table, scale);
}
