#pragma once

// GEMM configuration class: Handles the compile-time computation of the kernel parameters.
template <typename T>
concept GemmConfig = requires {
    { T::BLK_M } -> std::same_as<const int64_t &>;
    { T::BLK_N } -> std::same_as<const int64_t &>;
    { T::BLK_K } -> std::same_as<const int64_t &>;
    { T::GroupSizeM } -> std::same_as<const int64_t &>;
    { T::NumThreads } -> std::same_as<const int64_t &>;

    typename T::LayoutA;  // Layout of A
    typename T::LayoutB;  // Layout of B
    typename T::LayoutC;  // Layout of C

    typename T::SmemLayoutA;  // smem layout for one block of A
    typename T::SmemLayoutB;  // smem layout for one block of B

    typename T::GmemCopyA;  // gmem -> smem copy operation for A
    typename T::GmemCopyB;  // gmem -> smem copy operation for B
    typename T::GmemCopyC;  // rmem -> gmem copy operation for C

    typename T::TiledMMA;   // MMA operation
    typename T::SmemCopyA;  // smem -> rmem copy operation for A
    typename T::SmemCopyB;  // smem -> rmem copy operation for B
};