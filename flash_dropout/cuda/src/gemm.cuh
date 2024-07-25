#pragma once

#include <cute/tensor.hpp>

namespace ct = cute;

// Define some useful aliases
using ct::_;
using ct::Int;
template <typename T>
using Gmem = ct::ViewEngine<ct::gmem_ptr<T *>>;
template <typename T>
using Smem = ct::ViewEngine<ct::smem_ptr<T *>>;

template <typename Config, typename LayoutBlkC>
struct SmemGemm {
   private:
    ct::Tensor<Gmem<ct::half_t>, LayoutBlkC> &C;
    typename Config::TiledMMA tiled_mma;
    typename Config::SmemCopyA smem_tiled_copy_A;
    typename Config::SmemCopyB smem_tiled_copy_B;
    typename Config::GmemCopyC gmem_copy_C;

    decltype(tiled_mma.get_thread_slice(0u)) thread_mma;
    decltype(thread_mma.partition_fragment_C(C)) C_frag;

   public:
    __device__ SmemGemm(ct::Tensor<Gmem<ct::half_t>, LayoutBlkC> &C_)
        : C(C_),
          thread_mma(tiled_mma.get_thread_slice(threadIdx.x)),
          C_frag(thread_mma.partition_fragment_C(C)) {
        ct::clear(C_frag);
    }

    // Perform Smem GEMM: C += A @ B
    __device__ void operator()(
        const ct::Tensor<Smem<ct::half_t>, typename Config::SmemLayoutA> &sA,
        const ct::Tensor<Smem<ct::half_t>, typename Config::SmemLayoutB> &sB) {
        // Allocate registers distributed across threads to store operands
        auto A_frag = thread_mma.partition_fragment_A(sA);
        auto B_frag = thread_mma.partition_fragment_B(sB);

        // Load A and B from smem to registers (distributed across threads)
        auto thr_copy_A = smem_tiled_copy_A.get_thread_slice(threadIdx.x);
        auto sA_to_rA_src = thr_copy_A.partition_S(sA);   // COPY_V, COPY_M, COPY_K
        auto sA_to_rA_dst = thr_copy_A.retile_D(A_frag);  // COPY_V, COPY_M, COPY_K
        ct::copy(smem_tiled_copy_A, sA_to_rA_src, sA_to_rA_dst);

        auto thr_copy_B = smem_tiled_copy_B.get_thread_slice(threadIdx.x);
        auto sB_to_rB_src = thr_copy_B.partition_S(sB);   // COPY_V, COPY_N, COPY_K
        auto sB_to_rB_dst = thr_copy_B.retile_D(B_frag);  // COPY_V, COPY_N, COPY_K
        ct::copy(smem_tiled_copy_B, sB_to_rB_src, sB_to_rB_dst);

        // Perform GEMM
        ct::gemm(tiled_mma, A_frag, B_frag, C_frag);

        // Wait until all threads have finished using sA and sB
        __syncthreads();
    }

    // Write back result to gmem
    __device__ void write_back() {
        auto C_frag_out = thread_mma.partition_C(C);  // Corresponding location in output tensor
        ct::copy(gmem_copy_C, C_frag, C_frag_out);
        ct::cp_async_wait<0>();
    }
};

template <typename T, typename SrcLayout, typename DstLayout, typename TiledCopy>
__device__ void load_block_from_gmem_to_smem(const ct::Tensor<Gmem<T>, SrcLayout> &src,
                                             const ct::Tensor<Smem<T>, DstLayout> &dst,
                                             const TiledCopy &tiled_copy) {
    auto thread_copy = tiled_copy.get_thread_slice(threadIdx.x);
    auto src_frag = thread_copy.partition_S(src);
    auto dst_frag = thread_copy.partition_D(dst);
    ct::copy(tiled_copy, src_frag, dst_frag);
    ct::cp_async_wait<0>();
}

__device__ std::tuple<int64_t, int64_t> threadblock_swizzle(int64_t idx, int64_t m, int64_t n,
                                                            int64_t group_size_m) {
    // Reordering the block access pattern helps to improve L2 cache hit rate.
    // Triton's doc for matmul has a nice explanation:
    // https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html For m =
    // 3, n = 4, group_size_m = 2, produces the coordiantes in the following order:
    //  |  1 |  3 |  5 |  7 |
    //  |  2 |  4 |  6 |  8 |
    //  |  9 | 10 | 11 | 12 |
    int64_t blocks_per_group = group_size_m * n;
    int64_t first_block_idx_m = (idx / blocks_per_group) * group_size_m;
    group_size_m = min(m - first_block_idx_m,
                       group_size_m);  // Min to handle edge case of m % group_size_m != 0
    int64_t block_idx_m = first_block_idx_m + (idx % group_size_m);
    int64_t block_idx_n = (idx % blocks_per_group) / group_size_m;
    return std::make_tuple(block_idx_m, block_idx_n);
}

// Main kernel
template <typename Config, typename LayoutA, typename LayoutB, typename LayoutC>
__global__ void gemm_kernel(ct::Tensor<Gmem<ct::half_t>, LayoutA> A,
                            ct::Tensor<Gmem<ct::half_t>, LayoutB> B,
                            ct::Tensor<Gmem<ct::half_t>, LayoutC> C) {
    // Threadblock-level paratitioning
    auto [block_idx_m, block_idx_n] =
        threadblock_swizzle(blockIdx.x, ct::size<0>(A) / Config::BLK_M,
                            ct::size<0>(B) / Config::BLK_N, Config::GroupSizeM);
    auto block_shape_A = ct::Shape<Int<Config::BLK_M>, Int<Config::BLK_K>>{};
    auto block_shape_B = ct::Shape<Int<Config::BLK_N>, Int<Config::BLK_K>>{};
    auto block_shape_C = ct::Shape<Int<Config::BLK_M>, Int<Config::BLK_N>>{};
    auto A_blk =
        ct::local_tile(A, block_shape_A, ct::make_coord(block_idx_m, _));  // BLK_M, BLK_K, N_BLK_K
    auto B_blk =
        ct::local_tile(B, block_shape_B, ct::make_coord(block_idx_n, _));  // BLK_N, BLK_K, N_BLK_K
    auto C_blk =
        ct::local_tile(C, block_shape_C, ct::make_coord(block_idx_m, block_idx_n));  // BLK_M, BLK_N

    // Allocate shared memory for the operands
    typename Config::SmemLayoutA smem_layout_A;
    typename Config::SmemLayoutB smem_layout_B;
    __shared__ __align__(16) ct::half_t sA_data[ct::cosize_v<decltype(smem_layout_A)>];
    __shared__ __align__(16) ct::half_t sB_data[ct::cosize_v<decltype(smem_layout_B)>];
    auto sA = ct::make_tensor(ct::make_smem_ptr(sA_data), smem_layout_A);
    auto sB = ct::make_tensor(ct::make_smem_ptr(sB_data), smem_layout_B);

    // Main loop
    typename Config::GmemCopyA gmem_copy_A;
    typename Config::GmemCopyB gmem_copy_B;
    SmemGemm<Config, std::decay_t<decltype(C_blk.layout())>> smem_gemm(C_blk);
    for (size_t k = 0; k < ct::size<2>(A_blk); k++) {
        // Load the k-th A block from gmem to smem
        load_block_from_gmem_to_smem(A_blk(_, _, k), sA, gmem_copy_A);
        // Load the k-th B block from gmem to smem
        load_block_from_gmem_to_smem(B_blk(_, _, k), sB, gmem_copy_B);
        // Wait until all threads have finished loading A and B
        __syncthreads();
        smem_gemm(sA, sB);
    }
    smem_gemm.write_back();
}

// Host interface
template <typename Config, typename LayoutA, typename LayoutB, typename LayoutC>
void gemm(const ct::Tensor<Gmem<ct::half_t>, LayoutA> &A,
          const ct::Tensor<Gmem<ct::half_t>, LayoutB> &B,
          const ct::Tensor<Gmem<ct::half_t>, LayoutC> &C) {
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
    dim3 block_dim((M / Config::BLK_M) * (N / Config::BLK_N));
    dim3 thread_dim(Config::NumThreads);

    gemm_kernel<Config><<<block_dim, thread_dim>>>(A, B, C);
}