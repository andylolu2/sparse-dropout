#include <cute/tensor.hpp>
//
#include <curand_kernel.h>
#include <cutlass/util/device_dump.h>

#include <cute/arch/copy.hpp>
#include <cute/arch/mma_sm75.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/numeric/integral_constant.hpp>

namespace ct = cute;

using ct::_;
using ct::Int;

using RowMajor2D = ct::Layout<ct::Shape<int64_t, int64_t>, ct::Stride<int64_t, Int<1>>>;

template <typename scalar_t>
using Gmem = ct::ViewEngine<ct::gmem_ptr<scalar_t *>>;

template <typename scalar_t>
using Smem = ct::ViewEngine<ct::smem_ptr<scalar_t *>>;

template <typename scalar_t>
struct KernelTraits {
    using T = scalar_t;

   public:
    static constexpr int BLK_M = 64;
    static constexpr int BLK_N = 64;
    static constexpr int BLK_K = 32;
    static constexpr int NumThreads = 128;
    static constexpr int GroupSizeM = 2;

   private:
    static constexpr int AccessSizeBits = ct::sizeof_bits_v<ct::uint128_t>;
    static constexpr int ElemsPerLoad = AccessSizeBits / ct::sizeof_bits_v<scalar_t>;
    static constexpr int SmemAtomInner = 32;
    static constexpr int SmemAtomOuter = ElemsPerLoad;
    static constexpr int ThreadsPerRow = SmemAtomInner / ElemsPerLoad;

    using SmemAtomLayout = decltype(ct::composition(
        ct::Swizzle<2, 3, 3>{}, ct::Layout<ct::Shape<Int<SmemAtomOuter>, Int<SmemAtomInner>>,
                                           ct::Stride<Int<SmemAtomInner>, Int<1>>>{}));
    using SmemShapeA = ct::Shape<Int<BLK_M>, Int<BLK_K>>;
    using SmemShapeB = ct::Shape<Int<BLK_N>, Int<BLK_K>>;

    using GmemCopyAtom =
        ct::Copy_Atom<ct::AutoVectorizingCopyWithAssumedAlignment<AccessSizeBits>, scalar_t>;
    using GmemCopyThreadLayout =
        ct::Layout<ct::Shape<Int<NumThreads / ThreadsPerRow>, Int<ThreadsPerRow>>,
                   ct::Stride<Int<ThreadsPerRow>, Int<1>>>;
    using GmemCopyValLayout = ct::Layout<ct::Shape<Int<1>, Int<ElemsPerLoad>>>;

    using SmemCopyAtom = ct::Copy_Atom<ct::SM75_U32x4_LDSM_N, scalar_t>;

    using MmaAtom = ct::MMA_Atom<ct::SM75_16x8x8_F32F16F16F32_TN>;
    using MmaThreadLayout = ct::Layout<ct::Shape<Int<2>, Int<2>, Int<1>>>;
    using MmaValLayout = ct::Layout<ct::Shape<Int<1>, Int<2>, Int<2>>>;
    CUTE_STATIC_ASSERT(ct::size_v<MmaThreadLayout> *ct::size_v<MmaAtom::ThrID> == NumThreads);

   public:
    using BlockShapeA = ct::Shape<Int<BLK_M>, Int<BLK_K>>;
    using BlockShapeB = ct::Shape<Int<BLK_N>, Int<BLK_K>>;
    using BlockShapeC = ct::Shape<Int<BLK_M>, Int<BLK_N>>;

    using SmemLayoutA = decltype(ct::tile_to_shape(SmemAtomLayout{}, SmemShapeA{}));
    using SmemLayoutB = decltype(ct::tile_to_shape(SmemAtomLayout{}, SmemShapeB{}));

    using TiledMMA = ct::TiledMMA<MmaAtom, MmaThreadLayout, MmaValLayout>;

    using GmemTiledCopy =
        decltype(ct::make_tiled_copy(GmemCopyAtom{}, GmemCopyThreadLayout{}, GmemCopyValLayout{}));

    using SmemTiledCopyA = decltype(ct::make_tiled_copy_A(SmemCopyAtom{}, TiledMMA{}));
    using SmemTiledCopyB = decltype(ct::make_tiled_copy_B(SmemCopyAtom{}, TiledMMA{}));
};

template <typename KernelTraits, typename GmemCopyAtom, typename LayoutGaSrc, typename LayoutGaDst,
          typename LayoutGbSrc, typename LayoutGbDst, typename LayoutSa, typename LayoutSb,
          typename LayoutC, typename scalar_t = typename KernelTraits::T>
__device__ void matmul_thread(ct::Tensor<Gmem<scalar_t>, LayoutGaSrc> gA_to_sA_src,
                              ct::Tensor<Smem<scalar_t>, LayoutGaDst> gA_to_sA_dst,
                              ct::Tensor<Gmem<scalar_t>, LayoutGbSrc> gB_to_sB_src,
                              ct::Tensor<Smem<scalar_t>, LayoutGbDst> gB_to_sB_dst,
                              GmemCopyAtom gmem_copy_atom, ct::Tensor<Smem<scalar_t>, LayoutSa> sA,
                              ct::Tensor<Smem<scalar_t>, LayoutSb> sB,
                              ct::Tensor<Gmem<scalar_t>, LayoutC> C_blk) {
    typename KernelTraits::TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto rA = thr_mma.partition_fragment_A(sA);     // MMA, MMA_M, MMA_K
    auto rB = thr_mma.partition_fragment_B(sB);     // MMA, MMA_N, MMA_K
    auto rC = thr_mma.partition_fragment_C(C_blk);  // MMA, MMA_M, MMA_N
    auto gC = thr_mma.partition_C(C_blk);           // Corresponding fragment in gmem to write back
    ct::clear(rC);

    typename KernelTraits::SmemTiledCopyA tiled_copy_A;
    auto thr_copy_A = tiled_copy_A.get_thread_slice(threadIdx.x);
    auto sA_to_rA_src = thr_copy_A.partition_S(sA);  // COPY_V, COPY_M, COPY_K
    auto sA_to_rA_dst = thr_copy_A.retile_D(rA);     // COPY_V, COPY_M, COPY_K

    typename KernelTraits::SmemTiledCopyB tiled_copy_B;
    auto thr_copy_B = tiled_copy_B.get_thread_slice(threadIdx.x);
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

    int N_BLK_K = ct::size<3>(gA_to_sA_src);

#pragma unroll
    for (size_t k_blk = 0; k_blk < N_BLK_K; k_blk++) {
        ct::copy(gmem_copy_atom, gA_to_sA_src(_, _, _, k_blk), gA_to_sA_dst);
        ct::copy(gmem_copy_atom, gB_to_sB_src(_, _, _, k_blk), gB_to_sB_dst);
        __syncthreads();

        // Load rA and rB (distributed across registers of threads) by copying from smem to gmem
        ct::copy(tiled_copy_A, sA_to_rA_src, sA_to_rA_dst);
        ct::copy(tiled_copy_B, sB_to_rB_src, sB_to_rB_dst);
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

    // Write back result
    ct::copy(gmem_copy_atom, rC, gC);
}

template <typename KernelTraits, typename LayoutA, typename LayoutB, typename LayoutC,
          typename scalar_t = typename KernelTraits::T>
__device__ void matmul_threadblock(ct::Tensor<Gmem<scalar_t>, LayoutA> A_blk,
                                   ct::Tensor<Gmem<scalar_t>, LayoutB> B_blk,
                                   ct::Tensor<Gmem<scalar_t>, LayoutC> C_blk) {
    typename KernelTraits::SmemLayoutA smem_layout_A;
    typename KernelTraits::SmemLayoutB smem_layout_B;

    __shared__ scalar_t sA_data[ct::cosize_v<decltype(smem_layout_A)>];
    __shared__ scalar_t sB_data[ct::cosize_v<decltype(smem_layout_B)>];
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

    typename KernelTraits::GmemTiledCopy gmem_tiled_copy;
    auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(threadIdx.x);

    // Fragments for gmem -> smem copy
    auto gA_to_sA_src = gmem_thr_copy.partition_S(A_blk);  // COPY_V, COPY_M, COPY_K, N_BLK_K
    auto gA_to_sA_dst = gmem_thr_copy.partition_D(sA);     // COPY_V, COPY_M, COPY_K
    auto gB_to_sB_src = gmem_thr_copy.partition_S(B_blk);  // COPY_V, COPY_N, COPY_K, N_BLK_K
    auto gB_to_sB_dst = gmem_thr_copy.partition_D(sB);     // COPY_V, COPY_N, COPY_K

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

    matmul_thread<KernelTraits>(gA_to_sA_src, gA_to_sA_dst, gB_to_sB_src, gB_to_sB_dst,
                                gmem_tiled_copy, sA, sB, C_blk);
}

template <typename KernelTraits, typename scalar_t = typename KernelTraits::T>
__global__ void matmul_kernel(ct::Tensor<Gmem<scalar_t>, RowMajor2D> A,
                              ct::Tensor<Gmem<scalar_t>, RowMajor2D> B,
                              ct::Tensor<Gmem<scalar_t>, RowMajor2D> C) {
    using BlockShapeA = typename KernelTraits::BlockShapeA;
    using BlockShapeB = typename KernelTraits::BlockShapeB;
    using BlockShapeC = typename KernelTraits::BlockShapeC;

    auto A_blk_all = ct::tiled_divide(A, BlockShapeA{});  // (BLK_M, BLK_K), N_BLK_M, N_BLK_K
    auto B_blk_all = ct::tiled_divide(B, BlockShapeB{});  // (BLK_N, BLK_K), N_BLK_N, N_BLK_K
    auto C_blk_all = ct::tiled_divide(C, BlockShapeC{});  // (BLK_M, BLK_N), N_BLK_M, N_BLK_N

    int N_BLK_M = ct::size<1>(A_blk_all);
    int N_BLK_N = ct::size<1>(B_blk_all);
    int blocks_per_group = KernelTraits::GroupSizeM * N_BLK_N;
    int first_block_idx_m = (blockIdx.x / blocks_per_group) * KernelTraits::GroupSizeM;
    int group_size_m = min(N_BLK_M - first_block_idx_m, KernelTraits::GroupSizeM);  // Handle edge
    int block_idx_m = first_block_idx_m + (blockIdx.x % group_size_m);
    int block_idx_n = (blockIdx.x % blocks_per_group) / group_size_m;

    auto A_blk = ct::flatten(A_blk_all(_, block_idx_m, _));            // BLK_M, BLK_K, N_BLK_K
    auto B_blk = ct::flatten(B_blk_all(_, block_idx_n, _));            // BLK_N, BLK_K, N_BLK_K
    auto C_blk = ct::flatten(C_blk_all(_, block_idx_m, block_idx_n));  // BLK_M, BLK_N

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

    matmul_threadblock<KernelTraits>(A_blk, B_blk, C_blk);
}

template <typename scalar_t>
void matmul(ct::Tensor<Gmem<scalar_t>, RowMajor2D> A, ct::Tensor<Gmem<scalar_t>, RowMajor2D> B,
            ct::Tensor<Gmem<scalar_t>, RowMajor2D> C) {
    assert(ct::size<0>(A) == ct::size<0>(C));  // M
    assert(ct::size<1>(B) == ct::size<0>(C));  // N
    assert(ct::size<1>(A) == ct::size<1>(B));  // K

    using KernelTraits = KernelTraits<ct::half_t>;

    int64_t M = ct::size<0>(A);
    int64_t N = ct::size<0>(B);
    int64_t K = ct::size<1>(A);
    int64_t BLK_M = KernelTraits::BLK_M;
    int64_t BLK_N = KernelTraits::BLK_N;
    int64_t BLK_K = KernelTraits::BLK_K;

    // We don't handle predication yet
    assert(M % BLK_M == 0);
    assert(N % BLK_N == 0);
    assert(K % BLK_K == 0);

    dim3 block_dim((M / BLK_M) * (N / BLK_N));
    dim3 thread_dim(KernelTraits::NumThreads);

    matmul_kernel<KernelTraits><<<block_dim, thread_dim>>>(A, B, C);
}
