#include <cute/tensor.hpp>
//
#include <cutlass/util/device_dump.h>

#include <cute/arch/copy.hpp>
#include <cute/arch/mma_sm75.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/numeric/integral_constant.hpp>

namespace ct = cute;

using ct::_;
using ct::Int;

// TODO: Make the rank a template parameter

using DynStride2D = ct::Stride<int64_t, int64_t>;
using DynShape2D = ct::Shape<int64_t, int64_t>;
using Layout2D = ct::Layout<DynShape2D, DynStride2D>;

using DynStride3D = ct::Stride<int64_t, int64_t, int64_t>;

// template <typename scalar_t>
// using TensorGmem2d = ct::Tensor<ct::ViewEngine<ct::gmem_ptr<scalar_t *>>, Layout2D>;

template <typename scalar_t>
using Gmem = ct::ViewEngine<ct::gmem_ptr<scalar_t *>>;

template <typename scalar_t>
using Smem = ct::ViewEngine<ct::smem_ptr<scalar_t *>>;

template <typename scalar_t, int BLK_M, int BLK_N, int BLK_K, typename Thr_copy,
          typename gA_tile_Layout, typename sA_tile_Layout, typename gB_tile_Layout,
          typename sB_tile_Layout, typename sA_layout, typename sB_layout>
__device__ void matmul_warp(
    ct::Tensor<Gmem<scalar_t>, gA_tile_Layout> gA_to_sA_src,
    ct::Tensor<Smem<scalar_t>, sA_tile_Layout> gA_to_sA_dst,
    ct::Tensor<Gmem<scalar_t>, gB_tile_Layout> gB_to_sB_src,
    ct::Tensor<Smem<scalar_t>, sB_tile_Layout> gB_to_sB_dst, Thr_copy copy_atom,
    ct::Tensor<Smem<scalar_t>, sA_layout> sA, ct::Tensor<Smem<scalar_t>, sB_layout> sB,
    ct::Tensor<Gmem<scalar_t>, ct::Layout<ct::Shape<Int<BLK_M>, Int<BLK_N>>, DynStride2D>> C_blk) {
    using MMA_Atom = ct::MMA_Atom<ct::SM75_16x8x8_F32F16F16F32_TN>;
    using Warp_Shape = ct::Shape<Int<2>, Int<2>, Int<1>>;  // 2x2x1 warps = 128 threads
    using Val_Shape = ct::Shape<Int<1>, Int<2>, Int<2>>;
    using Tiled_MMA = ct::TiledMMA<MMA_Atom, ct::Layout<Warp_Shape>, ct::Layout<Val_Shape>>;
    using Smem_Copy_Atom = ct::Copy_Atom<ct::SM75_U32x4_LDSM_N, scalar_t>;

    Tiled_MMA tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto rA = thr_mma.partition_fragment_A(sA);     // MMA, MMA_M, MMA_K
    auto rB = thr_mma.partition_fragment_B(sB);     // MMA, MMA_N, MMA_K
    auto rC = thr_mma.partition_fragment_C(C_blk);  // MMA, MMA_M, MMA_N
    auto gC = thr_mma.partition_C(C_blk);           // MMA, MMA_M, MMA_N
    ct::clear(rC);

    auto tiled_copy_A = ct::make_tiled_copy_A(Smem_Copy_Atom{}, tiled_mma);
    auto thr_copy_A = tiled_copy_A.get_thread_slice(threadIdx.x);
    auto sA_to_rA_src = thr_copy_A.partition_S(sA);
    auto sA_to_rA_dst = thr_copy_A.retile_D(rA);

    auto tiled_copy_B = ct::make_tiled_copy_B(Smem_Copy_Atom{}, tiled_mma);
    auto thr_copy_B = tiled_copy_B.get_thread_slice(threadIdx.x);
    auto sB_to_rB_src = thr_copy_B.partition_S(sB);
    auto sB_to_rB_dst = thr_copy_B.retile_D(rB);

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
        ct::print(tiled_copy_A);
        ct::print(" <- tiled_copy_A\n");
        ct::print(sA);
        ct::print(" <- sA\n");
        ct::print(sA_to_rA_src);
        ct::print(" <- sA_to_rA_src\n");
        ct::print(sA_to_rA_dst);
        ct::print(" <- sA_to_rA_dst\n");
        ct::print(tiled_copy_B);
        ct::print(" <- tiled_copy_B\n");
        ct::print(sB);
        ct::print(" <- sB\n");
        ct::print(sB_to_rB_src);
        ct::print(" <- sB_to_rB_src\n");
        ct::print(sB_to_rB_dst);
        ct::print(" <- sB_to_rB_dst\n");
    }
#endif

    auto N_BLK_K = ct::size<3>(gA_to_sA_src);
    for (size_t k_blk = 0; k_blk < N_BLK_K; k_blk++) {
        // Populate sA and sB by copying each thread's fragment from gmem to smem
        ct::copy(copy_atom, gA_to_sA_src(_, _, _, k_blk), gA_to_sA_dst);
        ct::copy(copy_atom, gB_to_sB_src(_, _, _, k_blk), gB_to_sB_dst);
        __syncthreads();
        // Now sA and sB have data for this iteration
        // cutlass::debug::dump_shmem(sA.data().get(), sA.size());

        // Populate rA and rB (distributed across registers of threads) by copying from sA and sB
        ct::copy(tiled_copy_A, sA_to_rA_src, sA_to_rA_dst);
        ct::copy(tiled_copy_B, sB_to_rB_src, sB_to_rB_dst);
#if 0
        if (ct::thread0()) {
            for (size_t i = 0; i < ct::size(rA); i++) {
                printf("%.0f ", static_cast<float>(rA(i)));
            }
            printf("\n");
        }
#endif
        // Perform gemm on rA, rB, rC (inner loop handled by cute)
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

    // Write back C to gmem
    ct::copy(copy_atom, rC, gC);
}

template <typename scalar_t, int BLK_M, int BLK_N, int BLK_K>
__device__ void matmul_tb(
    ct::Tensor<Gmem<scalar_t>, ct::Layout<ct::Shape<Int<BLK_M>, Int<BLK_K>, int64_t>, DynStride3D>>
        A_blk,
    ct::Tensor<Gmem<scalar_t>, ct::Layout<ct::Shape<Int<BLK_N>, Int<BLK_K>, int64_t>, DynStride3D>>
        B_blk,
    ct::Tensor<Gmem<scalar_t>, ct::Layout<ct::Shape<Int<BLK_M>, Int<BLK_N>>, DynStride2D>> C_blk) {
    // using N_stages = ct::_1;

    __shared__ scalar_t smem_A_tb_data[BLK_M * BLK_K];
    __shared__ scalar_t smem_B_tb_data[BLK_N * BLK_K];

    // auto smem_layout_atom = ct::composition(
    //     ct::Swizzle<2, 3, 3>{},
    //     ct::make_layout(ct::make_shape(Int<8>{}, Int<32>{}), ct::make_stride(Int<32>{},
    //     Int<1>{})));
    // auto smem_layout_A =
    //     ct::tile_to_shape(smem_layout_atom, ct::make_shape(Int<BLK_M>{}, Int<BLK_K>{}));
    // auto smem_layout_B =
    //     ct::tile_to_shape(smem_layout_atom, ct::make_shape(Int<BLK_N>{}, Int<BLK_K>{}));
    auto smem_layout_A = ct::make_layout(ct::make_shape(Int<BLK_M>{}, Int<BLK_K>{}),
                                         ct::make_stride(Int<BLK_K>{}, Int<1>{}));
    auto smem_layout_B = ct::make_layout(ct::make_shape(Int<BLK_N>{}, Int<BLK_K>{}),
                                         ct::make_stride(Int<BLK_K>{}, Int<1>{}));

    auto sA = ct::make_tensor(ct::make_smem_ptr(smem_A_tb_data), smem_layout_A);
    auto sB = ct::make_tensor(ct::make_smem_ptr(smem_B_tb_data), smem_layout_B);

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

    // Fragments for gmem -> smem copy
    using Copy_Atom = ct::Copy_Atom<ct::DefaultCopy, scalar_t>;
    auto copy_atom = Copy_Atom{};
    auto tiled_copy_A =
        ct::make_tiled_copy(copy_atom, ct::make_layout(ct::make_shape(ct::_16{}, ct::_8{})),
                            ct::make_layout(ct::make_shape(ct::_1{}, ct::_1{})));
    auto thr_copy_A = tiled_copy_A.get_thread_slice(threadIdx.x);

    auto gA_to_sA_src = thr_copy_A.partition_S(A_blk);  // (Atom_V, Atom_X, Atom_Y, N_BLK_K)
    auto gA_to_sA_dst = thr_copy_A.partition_D(sA);     // (Atom_V, Atom_X, Atom_Y)
    auto gB_to_sB_src = thr_copy_A.partition_S(B_blk);  // (Atom_V, Atom_X, Atom_Y, N_BLK_K)
    auto gB_to_sB_dst = thr_copy_A.partition_D(sB);     // (Atom_V, Atom_X, Atom_Y)

#if 0
    if (ct::thread0()) {
        ct::print(tiled_copy_A);
        ct::print(" <- tiled_copy_A\n");
        ct::print(sA);
        ct::print(" <- sA\n");
        ct::print(gA_to_sA_dst);
        ct::print(" <- gA_to_sA_dst\n");
        ct::print(A_blk);
        ct::print(" <- A_blk\n");
        ct::print(gA_to_sA_src);
        ct::print(" <- gA_to_sA_src\n");
        ct::print(sB);
        ct::print(" <- sB\n");
        ct::print(gB_to_sB_dst);
        ct::print(" <- gB_to_sB_dst\n");
        ct::print(B_blk);
        ct::print(" <- B_blk\n");
        ct::print(gB_to_sB_src);
    }
#endif

    matmul_warp<scalar_t, BLK_M, BLK_N, BLK_K>(gA_to_sA_src, gA_to_sA_dst, gB_to_sB_src,
                                               gB_to_sB_dst, copy_atom, sA, sB, C_blk);
}

template <typename scalar_t>
__global__ void matmul(ct::Tensor<Gmem<scalar_t>, Layout2D> A,
                       ct::Tensor<Gmem<scalar_t>, Layout2D> B,
                       ct::Tensor<Gmem<scalar_t>, Layout2D> C) {
    static constexpr int BLK_M = 64;
    static constexpr int BLK_N = 32;
    static constexpr int BLK_K = 32;

    auto A_blk_all = ct::tiled_divide(
        A, ct::make_shape(Int<BLK_M>{}, Int<BLK_K>{}));  // (BLK_M, BLK_K), N_BLK_M, N_BLK_K
    auto B_blk_all = ct::tiled_divide(
        B, ct::make_shape(Int<BLK_N>{}, Int<BLK_K>{}));  // (BLK_N, BLK_K), N_BLK_N, N_BLK_K
    auto C_blk_all = ct::tiled_divide(
        C, ct::make_shape(Int<BLK_M>{}, Int<BLK_N>{}));     // (BLK_M, BLK_N), N_BLK_M, N_BLK_N
    auto A_blk = ct::flatten(A_blk_all(_, blockIdx.x, _));  // BLK_M, BLK_K, N_BLK_K
    auto B_blk = ct::flatten(B_blk_all(_, blockIdx.y, _));  // BLK_N, BLK_K, N_BLK_K
    auto C_blk = ct::flatten(C_blk_all(_, blockIdx.x, blockIdx.y));  // BLK_M, BLK_N

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
    }
#endif

    matmul_tb<scalar_t, BLK_M, BLK_N, BLK_K>(A_blk, B_blk, C_blk);

    // using BLK_K = ct::Int<32>;
    // using N_Stages = ct::Int<2>;

    // using SmemLayoutA = ct::Layout<ct::Shape<BLK_M, BLK_K, N_Stages>>;
    // using SmemLayoutB = ct::Layout<ct::Shape<BLK_N, BLK_K, N_Stages>>;
    // using SmemLayoutC = ct::Layout<ct::Shape<BLK_M, BLK_N>>;

    // auto gA = ct::tiled_divide(A, ct::Shape<BLK_M, BLK_K>{});  // (BLK_M, BLK_K), N_BLK_M,
    // N_BLK_K auto gB = ct::tiled_divide(B, ct::Shape<BLK_N, BLK_K>{});  // (BLK_N, BLK_K),
    // N_BLK_N, N_BLK_K auto gC = ct::tiled_divide(C, ct::Shape<BLK_M, BLK_N>{});  // (BLK_M,
    // BLK_N), N_BLK_M, N_BLK_N

    // __shared__ scalar_t smem_A[ct::cosize_v<SmemLayoutA>];
    // __shared__ scalar_t smem_B[ct::cosize_v<SmemLayoutB>];
    // __shared__ scalar_t smem_C[ct::cosize_v<SmemLayoutC>];
    // auto sA = ct::make_tensor(ct::make_smem_ptr(smem_A), SmemLayoutA{});
    // auto sB = ct::make_tensor(ct::make_smem_ptr(smem_B), SmemLayoutB{});
    // auto sC = ct::make_tensor(ct::make_smem_ptr(smem_C), SmemLayoutC{});

    // if (ct::thread0()) {
    //     ct::print(gA);
    //     ct::print("<- gA\n");
    //     ct::print(gB);
    //     ct::print("<- gB\n");
    //     ct::print(gC);
    //     ct::print("<- gC\n");

    //     ct::print(sA);
    //     ct::print("<- sA\n");
    //     ct::print(sB);
    //     ct::print("<- sB\n");
    //     ct::print(sC);
    //     ct::print("<- sC\n");
    // }

    // auto mma_atom = ct::MMA_Atom<ct::SM75_16x8x8_F32F16F16F32_TN>{};
    // auto tiled_mma =
    //     ct::make_tiled_mma(mma_atom, ct::make_layout(ct::make_shape(ct::_1{}, ct::_1{},
    //     ct::_1{})),
    //                        ct::make_layout(ct::make_shape(ct::_2{}, ct::_2{}, ct::_1{})));

    // auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    // auto tCrA = thr_mma.partition_fragment_A(sA);  // (MMA_V), MMA_M, MMA_K, N_stages
    // auto tCrB = thr_mma.partition_fragment_B(sB);  // (MMA_V), MMA_N, MMA_K, N_stages
    // auto tCrC = thr_mma.partition_fragment_C(sC);  // (MMA_V), MMA_N, MMA_K, N_stages

    // auto smem_copy_atom = ct::Copy_Atom<ct::SM75_U32x1_LDSM_N, scalar_t>{};
    // auto tile_copy_A = ct::make_tiled_copy_A(smem_copy_atom, tiled_mma);
    // auto thr_copy_A = ct::make_tiled_copy_A(smem_copy_atom,
    // tiled_mma).get_slice(threadIdx.x); auto tCsA = thr_copy_A.partition_S(sA); auto
    // tCrA_copy_view = thr_copy_A.retile_D(tCrA);

    // auto thr_copy_B = ct::make_tiled_copy_B(smem_copy_atom,
    // tiled_mma).get_slice(threadIdx.x); auto tCsB = thr_copy_B.partition_S(sB); auto
    // tCrB_copy_view = thr_copy_B.retile_D(tCrB);

    // if (ct::thread0()) {
    //     ct::print(sA);
    //     ct::print("<- sA\n");
    //     ct::print(thr_mma);
    //     ct::print("<- thr_mma\n");
    //     ct::print(tCrA);
    //     ct::print("<- tCrA\n");
    //     ct::print(tCrA_copy_view);
    //     ct::print("<- tCrA_copy_vew\n");
    //     ct::print(tile_copy_A.tidfrg_S(sA));
    //     ct::print("<- tidfrg_S(sA)\n");
    //     ct::print(tCsA);
    //     ct::print("<- tCsA\n");
    //     ct::print(tCrB);
    //     ct::print("<- tCrB\n");
    //     ct::print(tCrB_copy_view);
    //     ct::print("<- tCrB_copy_vew\n");
    //     ct::print(tCsB);
    //     ct::print("<- tCsB\n");
    //     ct::print(tCrC);
    //     ct::print("<- tCrC\n");
    // }
    // // auto tCrC = thr_mma.partition_fragment_C(gC);

    // // auto copy_atom = ct::Copy_Atom<ct::SM75_U32x1_LDSM_N, scalar_t>{};
    // // auto thr_copy_A = ct::make_tiled_copy_A(copy_atom, tiled_mma).get_slice(threadIdx.x);
    // // auto tCsA = thr_copy_A.partition_S(sA);
    // // auto tCrA_copy_view = thr_copy_A.retile_D(tCrA);

    // // auto thr_copy_B = ct::make_tiled_copy_B(copy_atom, tiled_mma).get_slice(threadIdx.x);
    // // auto tCsB = thr_copy_B.partition_S(sB);
    // // auto tCrB_copy_view = thr_copy_B.retile_D(tCrB);

    // // for (size_t i = 0; i < ct::ceil_div(A.size(1), BLK_K{}); ++i) {
    // //     gmem_to_smem(gA(_, blockIdx.x, i), sA);
    // //     gmem_to_smem(gB(_, blockIdx.y, i), sB);

    // //     // ct::copy(ct::SM75_U32x1_LDSM_N{}, tCsA, tCrA_copy_view);
    // //     // ct::copy(ct::SM75_U32x1_LDSM_N{}, tCsB, tCrB_copy_view);

    // //     // ct::gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
    // // }

    // return;
}