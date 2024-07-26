#pragma once

#include <cute/tensor.hpp>

namespace ct = cute;
using ct::Int;

/* GEMM configuration class: Handles the compile-time computation of the kernel parameters.
 * Required parameters:
 * - BLK_M, BLK_N, BLK_K: The block sizes for the GEMM operation.
 * - GroupSizeM: The number of thread blocks in the M dimension.
 * - NumThreads: The number of threads per thread block.
 * - LayoutA, LayoutB, LayoutC: The layouts of A, B, and C.
 * - SmemLayoutA, SmemLayoutB: The layouts of the shared memory blocks for A and B.
 * - GmemCopyA, GmemCopyB, GmemCopyC: The copy operations for A, B, and C.
 * - TiledMMA: The MMA operation.
 * - SmemCopyA, SmemCopyB: The copy operations for A and B from shared memory to register memory.
 */
template <bool RowMajorA, bool RowMajorB>
struct GemmConfigImpl {
   public:
    // 128x128x64 blocks seems to be a good default
    static constexpr int64_t BLK_M = 128;
    static constexpr int64_t BLK_N = 128;
    static constexpr int64_t BLK_K = 64;
    static constexpr int64_t NumThreads = 128;  // 4 warps

   private:
    using RowMajorLayout = ct::Layout<ct::Shape<int64_t, int64_t>, ct::Stride<int64_t, Int<1>>>;
    using ColMajorLayout = ct::Layout<ct::Shape<int64_t, int64_t>, ct::Stride<Int<1>, int64_t>>;

    static constexpr int AccessSizeBits = 128;
    static constexpr int ElemsPerLoad = AccessSizeBits / ct::sizeof_bits_v<ct::half_t>;
    static constexpr int SmemAtomInner = std::min(64, static_cast<int>(BLK_K));
    static constexpr int SmemAtomOuter = ElemsPerLoad;
    static constexpr int ThreadsPerRow = SmemAtomInner / ElemsPerLoad;

    using BlockShapeA = ct::Shape<Int<BLK_M>, Int<BLK_K>>;
    using BlockShapeB = ct::Shape<Int<BLK_N>, Int<BLK_K>>;

    // The layout of one tile of the smem block, will be tiled to fill the entire block.
    // The choice of this layout is important for performance.
    // Swizzling reduces shared memory bank conflicts.
    using RowMajorSmemLayoutAtom = decltype(ct::composition(
        ct::Swizzle<3, 3, 3>{}, ct::Layout<ct::Shape<Int<SmemAtomOuter>, Int<SmemAtomInner>>,
                                           ct::Stride<Int<SmemAtomInner>, Int<1>>>{}));
    using ColMajorSmemLayoutAtom = decltype(ct::composition(
        ct::Swizzle<3, 3, 3>{}, ct::Layout<ct::Shape<Int<SmemAtomInner>, Int<SmemAtomOuter>>,
                                           ct::Stride<Int<1>, Int<SmemAtomInner>>>{}));

   public:
    // Layout of each block of A/B in shared memory
    using SmemLayoutA = decltype(ct::tile_to_shape(
        std::conditional_t<RowMajorA, RowMajorSmemLayoutAtom, ColMajorSmemLayoutAtom>{},
        BlockShapeA{}));
    using SmemLayoutB = decltype(ct::tile_to_shape(
        std::conditional_t<RowMajorB, RowMajorSmemLayoutAtom, ColMajorSmemLayoutAtom>{},
        BlockShapeB{}));

   private:
    using GmemCopyAtom = ct::Copy_Atom<ct::SM80_CP_ASYNC_CACHEALWAYS<ct::uint128_t>, ct::half_t>;
    using RowMajorGmemCopy = decltype(ct::make_tiled_copy(
        GmemCopyAtom{},
        ct::Layout<ct::Shape<Int<NumThreads / ThreadsPerRow>, Int<ThreadsPerRow>>,
                   ct::Stride<Int<ThreadsPerRow>, Int<1>>>{},
        ct::Layout<ct::Shape<Int<1>, Int<ElemsPerLoad>>>{}));
    using ColMajorGmemCopy = decltype(ct::make_tiled_copy(
        GmemCopyAtom{},
        ct::Layout<ct::Shape<Int<NumThreads / ThreadsPerRow>, Int<ThreadsPerRow>>,
                   ct::Stride<Int<1>, Int<NumThreads / ThreadsPerRow>>>{},
        ct::Layout<ct::Shape<Int<ElemsPerLoad>, Int<1>>>{}));

   public:
    using GmemCopyA = std::conditional_t<RowMajorA, RowMajorGmemCopy, ColMajorGmemCopy>;
    using GmemCopyB = std::conditional_t<RowMajorB, RowMajorGmemCopy, ColMajorGmemCopy>;

   private:
    // The atom of the smem -> rmem copy for A/B. Loads 4 8x8 matrices (distributed across threads)
    // at a time.
    using RowMajorSmemCopyAtom = ct::Copy_Atom<ct::SM75_U32x4_LDSM_N, ct::half_t>;
    using ColMajorSmemCopyAtom = ct::Copy_Atom<ct::SM75_U16x8_LDSM_T, ct::half_t>;

    // The atom for the MMA operation. Each atom is a warp-wise instruction that computes a 16x8x8
    // mma (with tensor cores).
    using MmaAtom = ct::MMA_Atom<ct::SM80_16x8x16_F32F16F16F32_TN>;
    // We have 128 threads, so we use 4 warps laid out in 2x2x1.
    using MmaAtomLayout = ct::Layout<ct::Shape<Int<2>, Int<2>, Int<1>>>;
    // We want to use the `ldmatrix.x4.m8n8` instruction which loads 4 8x8 matrices for maximum
    // efficiency. To make the operands A and B divisible into 4 8x8 matrices, we expand the problem
    // size for each warp to 16x16x16. Accounting for the fact that we use 4 warps laid out in
    // 2x2x1, the full tile size is 32x32x16.
    using MmaTiledShape = ct::Tile<Int<32>, Int<32>, Int<16>>;

   public:
    using TiledMMA = ct::TiledMMA<MmaAtom, MmaAtomLayout, MmaTiledShape>;
    using SmemCopyA = decltype(ct::make_tiled_copy_A(
        std::conditional_t<RowMajorA, RowMajorSmemCopyAtom, ColMajorSmemCopyAtom>{}, TiledMMA{}));
    using SmemCopyB = decltype(ct::make_tiled_copy_B(
        std::conditional_t<RowMajorB, RowMajorSmemCopyAtom, ColMajorSmemCopyAtom>{}, TiledMMA{}));
};