#pragma once

#include <torch/extension.h>

#include <cute/tensor.hpp>
#include <random>

namespace ct = cute;

// template <int BLK_M, int BLK_K, int BLK_MASK_M, int BLK_MASK_K, int BLK_MASK_TABLE_M,
//   int BLK_MASK_TABLE_K, int BLK_MASK_T_M, int BLK_MASK_T_K>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> make_mask(int64_t m, int64_t k,
                                                                           int64_t block_size,
                                                                           double p) {
    // static_assert(BLK_M % BLK_MASK_M == 0, "BLK_M must be divisible by BLK_MASK_M");
    // static_assert(BLK_K % BLK_MASK_K == 0, "BLK_N must be divisible by BLK_MASK_K");
    // static_assert(BLK_M % BLK_MASK_T_M == 0, "BLK_M must be divisible by BLK_MASK_T_M");
    // static_assert(BLK_K % BLK_MASK_T_K == 0, "BLK_N must be divisible by BLK_MASK_T_K");
    // static_assert(BLK_M % BLK_MASK_TABLE_M == 0, "BLK_M must be divisible by BLK_MASK_TABLE_M");
    // static_assert(BLK_K % BLK_MASK_TABLE_K == 0, "BLK_N must be divisible by BLK_MASK_TABLE_K");
    assert(m % block_size == 0 && "M must be divisible by block_size");
    assert(k % block_size == 0 && "K must be divisible by block_size");

    // static constexpr int mask_stride_m = BLK_M / BLK_MASK_M;
    // static constexpr int mask_stride_k = BLK_K / BLK_MASK_K;
    // static constexpr int mask_T_stride_m = BLK_M / BLK_MASK_T_M;
    // static constexpr int mask_T_stride_k = BLK_K / BLK_MASK_T_K;
    // static constexpr int mask_table_stride_m = BLK_M / BLK_MASK_TABLE_M;
    // static constexpr int mask_table_stride_k = BLK_K / BLK_MASK_TABLE_K;
    torch::Tensor mask = torch::zeros({m / block_size, k / block_size}, torch::kBool);
    torch::Tensor mask_T = torch::zeros({k / block_size, m / block_size}, torch::kBool);
    std::vector<std::array<int64_t, 2>> mask_table;

    // static constexpr int TMaskBits = 64;
    // torch::Tensor mask =
    //     torch::zeros({mask_stride_m * ct::ceil_div(M, BLK_M),
    //                   ct::ceil_div(mask_stride_k * ct::ceil_div(K, BLK_K), TMaskBits)},
    //                  torch::kInt64);
    // torch::Tensor mask_T =
    //     torch::zeros({mask_T_stride_k * ct::ceil_div(K, BLK_K),
    //                   ct::ceil_div(mask_T_stride_m * ct::ceil_div(M, BLK_M), TMaskBits)},
    //                  torch::kInt64);
    // torch::Tensor mask_table = torch::empty({mask_table_stride_m * mask_table_stride_k *
    //                                              ct::ceil_div(K, BLK_K) * ct::ceil_div(M, BLK_M),
    //                                          2},
    //                                         torch::kInt64);

    auto mask_a = mask.accessor<bool, 2>();
    auto mask_T_a = mask_T.accessor<bool, 2>();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0, 1);

    int64_t count = 0;
    for (int64_t i = 0; i < m / block_size; ++i) {
        for (int64_t j = 0; j < k / block_size; ++j) {
            bool drop = dis(gen) < p;

            if (drop) {
                mask_a[i][j] = true;
                mask_T_a[j][i] = true;
            } else {
                mask_table.emplace_back(std::array<int64_t, 2>{i, j});
                count++;
            }
        }
    }
    //     for (int64_t i = 0; i < ct::ceil_div(M, BLK_M); ++i) {
    //         for (int64_t j = 0; j < ct::ceil_div(K, BLK_K); ++j) {
    //             bool drop = dis(gen) < p;

    //             if (drop) {
    //                 uint64_t val = 1;
    // #pragma unroll
    //                 for (int64_t m = 0; m < mask_stride_m; ++m) {
    //                     auto m_idx = i * mask_stride_m + m;
    // #pragma unroll
    //                     for (int64_t k = 0; k < mask_stride_k; ++k) {
    //                         auto k_idx = j * mask_stride_k + k;
    //                         mask_a[m_idx][k_idx / TMaskBits] |= (val << (k_idx % TMaskBits));
    //                     }
    //                 }
    // #pragma unroll
    //                 for (int64_t m = 0; m < mask_T_stride_m; ++m) {
    //                     auto m_idx = i * mask_T_stride_m + m;
    // #pragma unroll
    //                     for (int64_t k = 0; k < mask_T_stride_k; ++k) {
    //                         auto k_idx = j * mask_T_stride_k + k;
    //                         mask_T_a[k_idx][m_idx / TMaskBits] |= (val << (m_idx % TMaskBits));
    //                     }
    //                 }
    //             } else {
    // #pragma unroll
    //                 for (int64_t m = 0; m < mask_table_stride_m; ++m) {
    //                     auto m_idx = i * mask_table_stride_m + m;
    // #pragma unroll
    //                     for (int64_t k = 0; k < mask_table_stride_k; ++k) {
    //                         auto k_idx = j * mask_table_stride_k + k;
    //                         mask_table_a[count][0] = m_idx;
    //                         mask_table_a[count][1] = k_idx;
    //                         count++;
    //                     }
    //                 }
    //             }
    //         }
    // }

    auto mask_cuda = mask.to(torch::kCUDA, true);
    auto mask_T_cuda = mask_T.to(torch::kCUDA, true);

    auto options = torch::TensorOptions().dtype(at::kLong);
    torch::Tensor mask_table_cuda =
        torch::from_blob(&mask_table[0][0], {count, 2}, options).clone().to(torch::kCUDA, true);

    return {mask_cuda, mask_T_cuda, mask_table_cuda, count};
}

template <int BLK_M, int BLK_K, int BLK_MASK_M, int BLK_MASK_K, int BLK_MASK_TABLE_M,
          int BLK_MASK_TABLE_K, int BLK_MASK_T_M, int BLK_MASK_T_K>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> make_mask_from_existing(
    torch::Tensor m) {
    static_assert(BLK_M % BLK_MASK_M == 0, "BLK_M must be divisible by BLK_MASK_M");
    static_assert(BLK_K % BLK_MASK_K == 0, "BLK_N must be divisible by BLK_MASK_K");
    static_assert(BLK_M % BLK_MASK_T_M == 0, "BLK_M must be divisible by BLK_MASK_T_M");
    static_assert(BLK_K % BLK_MASK_T_K == 0, "BLK_N must be divisible by BLK_MASK_T_K");
    static_assert(BLK_M % BLK_MASK_TABLE_M == 0, "BLK_M must be divisible by BLK_MASK_TABLE_M");
    static_assert(BLK_K % BLK_MASK_TABLE_K == 0, "BLK_N must be divisible by BLK_MASK_TABLE_K");

    static constexpr int mask_stride_m = BLK_M / BLK_MASK_M;
    static constexpr int mask_stride_k = BLK_K / BLK_MASK_K;
    static constexpr int mask_T_stride_m = BLK_M / BLK_MASK_T_M;
    static constexpr int mask_T_stride_k = BLK_K / BLK_MASK_T_K;
    static constexpr int mask_table_stride_m = BLK_M / BLK_MASK_TABLE_M;
    static constexpr int mask_table_stride_k = BLK_K / BLK_MASK_TABLE_K;

    static constexpr int TMaskBits = 64;
    int64_t N_BLK_M = m.size(0);
    int64_t N_BLK_K = m.size(1);
    torch::Tensor mask = torch::zeros(
        {mask_stride_m * N_BLK_M, ct::ceil_div(mask_stride_k * N_BLK_K, TMaskBits)}, torch::kInt64);
    torch::Tensor mask_T = torch::zeros(
        {mask_T_stride_k * N_BLK_K, ct::ceil_div(mask_T_stride_m * N_BLK_M, TMaskBits)},
        torch::kInt64);
    torch::Tensor mask_table = torch::empty(
        {mask_table_stride_m * mask_table_stride_k * N_BLK_M * N_BLK_K, 2}, torch::kInt64);

    auto m_a = m.accessor<bool, 2>();
    auto mask_a = mask.accessor<int64_t, 2>();
    auto mask_T_a = mask_T.accessor<int64_t, 2>();
    auto mask_table_a = mask_table.accessor<int64_t, 2>();

    int64_t count = 0;
    for (int64_t i = 0; i < N_BLK_M; ++i) {
        for (int64_t j = 0; j < N_BLK_K; ++j) {
            bool drop = m_a[i][j];
            if (drop) {
                uint64_t val = 1;
                for (int64_t m = 0; m < mask_stride_m; ++m) {
                    auto m_idx = i * mask_stride_m + m;
                    for (int64_t k = 0; k < mask_stride_k; ++k) {
                        auto k_idx = j * mask_stride_k + k;
                        mask_a[m_idx][k_idx / TMaskBits] |= (val << (k_idx % TMaskBits));
                    }
                }
                for (int64_t m = 0; m < mask_T_stride_m; ++m) {
                    auto m_idx = i * mask_T_stride_m + m;
                    for (int64_t k = 0; k < mask_T_stride_k; ++k) {
                        auto k_idx = j * mask_T_stride_k + k;
                        mask_T_a[k_idx][m_idx / TMaskBits] |= (val << (m_idx % TMaskBits));
                    }
                }
            } else {
                for (int64_t m = 0; m < mask_table_stride_m; ++m) {
                    auto m_idx = i * mask_table_stride_m + m;
                    for (int64_t k = 0; k < mask_table_stride_k; ++k) {
                        auto k_idx = j * mask_table_stride_k + k;
                        mask_table_a[count][0] = m_idx;
                        mask_table_a[count][1] = k_idx;
                        count++;
                    }
                }
            }
        }
    }

    auto mask_cuda = mask.to(torch::kCUDA, true);
    auto mask_T_cuda = mask_T.to(torch::kCUDA, true);
    auto mask_table_cuda = mask_table.to(torch::kCUDA, true);

    return {mask_cuda, mask_T_cuda, mask_table_cuda, count};
}