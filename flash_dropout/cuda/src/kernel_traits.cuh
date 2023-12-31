#pragma once

#include <cute/tensor.hpp>
//
#include <cute/arch/mma_sm75.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/numeric/integral_constant.hpp>

namespace ct = cute;

using ct::Int;

template <typename scalar_t, int BLK_M_, int BLK_N_, int BLK_K_, int GroupSizeM_, bool RowMajorA,
          bool RowMajorB, bool RowMajorC>
struct KernelTraits {
    using T = scalar_t;
    using TMask = ct::uint64_t;

   public:
    static constexpr int BLK_M = BLK_M_;
    static constexpr int BLK_N = BLK_N_;
    static constexpr int BLK_K = BLK_K_;
    static constexpr int NumWarps = 4;
    static constexpr int NumThreads = 32 * NumWarps;
    static constexpr int GroupSizeM = GroupSizeM_;
    using LayoutA = ct::Layout<
        ct::Shape<int64_t, int64_t>,
        std::conditional_t<RowMajorA, ct::Stride<int64_t, Int<1>>, ct::Stride<Int<1>, int64_t>>>;
    using LayoutB = ct::Layout<
        ct::Shape<int64_t, int64_t>,
        std::conditional_t<RowMajorB, ct::Stride<int64_t, Int<1>>, ct::Stride<Int<1>, int64_t>>>;
    // Only support row major C
    using LayoutC = ct::Layout<
        ct::Shape<int64_t, int64_t>,
        std::conditional_t<RowMajorC, ct::Stride<int64_t, Int<1>>, ct::Stride<Int<1>, int64_t>>>;
    using LayoutMask = ct::Layout<ct::Shape<int64_t, int64_t>, ct::Stride<int64_t, Int<1>>>;
    using LayoutMaskTable = ct::Layout<ct::Shape<int64_t, Int<2>>, ct::Stride<Int<2>, Int<1>>>;

   private:
    static constexpr int AccessSizeBits = ct::sizeof_bits_v<ct::uint128_t>;
    static constexpr int ElemsPerLoad = AccessSizeBits / ct::sizeof_bits_v<scalar_t>;
    static constexpr int SmemAtomInner = min(Int<32>{}, BLK_K);
    static constexpr int SmemAtomOuter = ElemsPerLoad;
    static constexpr int ThreadsPerRow = SmemAtomInner / ElemsPerLoad;

    using SmemCopyInstrA =
        std::conditional_t<RowMajorA, ct::SM75_U32x4_LDSM_N, ct::SM75_U16x8_LDSM_T>;
    using SmemCopyAtomA = ct::Copy_Atom<SmemCopyInstrA, scalar_t>;
    using SmemAtomLayoutA =
        ct::Layout<ct::Shape<Int<SmemAtomOuter>, Int<SmemAtomInner>>,
                   std::conditional_t<RowMajorA, ct::Stride<Int<SmemAtomInner>, Int<1>>,
                                      ct::Stride<Int<1>, Int<SmemAtomOuter>>>>;

    using SmemCopyInstrB =
        std::conditional_t<RowMajorB, ct::SM75_U32x4_LDSM_N, ct::SM75_U16x8_LDSM_T>;
    using SmemCopyAtomB = ct::Copy_Atom<SmemCopyInstrB, scalar_t>;
    using SmemAtomLayoutB =
        ct::Layout<ct::Shape<Int<SmemAtomOuter>, Int<SmemAtomInner>>,
                   std::conditional_t<RowMajorB, ct::Stride<Int<SmemAtomInner>, Int<1>>,
                                      ct::Stride<Int<1>, Int<SmemAtomOuter>>>>;

    using SmemAtomLayoutCanonical = ct::Layout<ct::Shape<Int<SmemAtomOuter>, Int<SmemAtomInner>>,
                                               ct::Stride<Int<SmemAtomInner>, Int<1>>>;

    using SmemAtomLayoutSwizzleA =
        decltype(ct::composition(ct::Swizzle<2, 3, 3>{}, SmemAtomLayoutA{}));
    using SmemAtomLayoutSwizzleB =
        decltype(ct::composition(ct::Swizzle<2, 3, 3>{}, SmemAtomLayoutB{}));
    using SmemAtomLayoutSwizzleCanonical =
        decltype(ct::composition(ct::Swizzle<2, 3, 3>{}, SmemAtomLayoutCanonical{}));
    using SmemShapeA = ct::Shape<Int<BLK_M>, Int<BLK_K>>;
    using SmemShapeB = ct::Shape<Int<BLK_N>, Int<BLK_K>>;

    using GmemCopyAtom =
        ct::Copy_Atom<ct::AutoVectorizingCopyWithAssumedAlignment<AccessSizeBits>, scalar_t>;

    using GmemCopyThreadLayoutRow =
        ct::Layout<ct::Shape<Int<NumThreads / ThreadsPerRow>, Int<ThreadsPerRow>>,
                   ct::Stride<Int<ThreadsPerRow>, Int<1>>>;
    using GmemCopyValLayoutRow = ct::Layout<ct::Shape<Int<1>, Int<ElemsPerLoad>>>;

    using GmemCopyThreadLayoutCol =
        ct::Layout<ct::Shape<Int<ThreadsPerRow>, Int<NumThreads / ThreadsPerRow>>,
                   ct::Stride<Int<1>, Int<ThreadsPerRow>>>;
    using GmemCopyValLayoutCol = ct::Layout<ct::Shape<Int<ElemsPerLoad>, Int<1>>>;

    using GmemCopyThreadLayoutA =
        std::conditional_t<RowMajorA, GmemCopyThreadLayoutRow, GmemCopyThreadLayoutCol>;
    using GmemCopyThreadLayoutB =
        std::conditional_t<RowMajorB, GmemCopyThreadLayoutRow, GmemCopyThreadLayoutCol>;
    using GmemCopyThreadLayoutC =
        std::conditional_t<RowMajorC, GmemCopyThreadLayoutRow, GmemCopyThreadLayoutCol>;
    using GmemCopyValLayoutA =
        std::conditional_t<RowMajorA, GmemCopyValLayoutRow, GmemCopyValLayoutCol>;
    using GmemCopyValLayoutB =
        std::conditional_t<RowMajorB, GmemCopyValLayoutRow, GmemCopyValLayoutCol>;
    using GmemCopyValLayoutC =
        std::conditional_t<RowMajorC, GmemCopyValLayoutRow, GmemCopyValLayoutCol>;
    CUTE_STATIC_ASSERT(
        ct::size_v<GmemCopyThreadLayoutA> *ct::size_v<typename GmemCopyAtom::ThrID> == NumThreads);
    CUTE_STATIC_ASSERT(
        ct::size_v<GmemCopyThreadLayoutB> *ct::size_v<typename GmemCopyAtom::ThrID> == NumThreads);

    using MmaAtom = ct::MMA_Atom<ct::SM75_16x8x8_F32F16F16F32_TN>;
    using MmaThreadLayout = ct::Layout<ct::Shape<Int<NumWarps>, Int<1>, Int<1>>>;
    using MmaValLayout = ct::Layout<ct::Shape<Int<1>, Int<2>, Int<2>>>;
    CUTE_STATIC_ASSERT(ct::size_v<MmaThreadLayout> *ct::size_v<MmaAtom::ThrID> == NumThreads);

   public:
    using BlockShapeA = ct::Shape<Int<BLK_M>, Int<BLK_K>>;
    using BlockShapeB = ct::Shape<Int<BLK_N>, Int<BLK_K>>;
    using BlockShapeC = ct::Shape<Int<BLK_M>, Int<BLK_N>>;

    using SmemLayoutA = decltype(ct::tile_to_shape(SmemAtomLayoutSwizzleA{}, SmemShapeA{}));
    using SmemLayoutB = decltype(ct::tile_to_shape(SmemAtomLayoutSwizzleB{}, SmemShapeB{}));
    using SmemLayoutACanonical =
        decltype(ct::tile_to_shape(SmemAtomLayoutSwizzleCanonical{}, SmemShapeA{}));
    using SmemLayoutBCanonical =
        decltype(ct::tile_to_shape(SmemAtomLayoutSwizzleCanonical{}, SmemShapeB{}));

    using TiledMMA = ct::TiledMMA<MmaAtom, MmaThreadLayout, MmaValLayout>;

    using GmemTiledCopyA = decltype(ct::make_tiled_copy(GmemCopyAtom{}, GmemCopyThreadLayoutA{},
                                                        GmemCopyValLayoutA{}));
    using GmemTiledCopyB = decltype(ct::make_tiled_copy(GmemCopyAtom{}, GmemCopyThreadLayoutB{},
                                                        GmemCopyValLayoutB{}));
    using GmemTiledCopyC = decltype(ct::make_tiled_copy(GmemCopyAtom{}, GmemCopyThreadLayoutC{},
                                                        GmemCopyValLayoutC{}));

    using SmemTiledCopyA = decltype(ct::make_tiled_copy_A(SmemCopyAtomA{}, TiledMMA{}));
    using SmemTiledCopyB = decltype(ct::make_tiled_copy_B(SmemCopyAtomB{}, TiledMMA{}));
};