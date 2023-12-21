#pragma once

#include <emmintrin.h>
#include <torch/extension.h>

#include <cute/tensor.hpp>
#include <random>

namespace ct = cute;

template <typename KernelTraits>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> make_mask(int64_t M, int64_t K,
                                                                           double p) {
    static constexpr int TMaskBits = 64;
    int64_t N_BLK_M = ct::ceil_div(M, KernelTraits::BLK_M);
    int64_t N_BLK_K = ct::ceil_div(K, KernelTraits::BLK_K);

    auto mask = torch::empty({N_BLK_M, ct::ceil_div(N_BLK_K, TMaskBits)}, torch::kInt64);
    auto mask_T = torch::empty({N_BLK_K, ct::ceil_div(N_BLK_M, TMaskBits)}, torch::kInt64);
    auto mask_table = torch::empty({N_BLK_K * N_BLK_M, 2}, torch::kInt64);

    auto mask_a = mask.accessor<int64_t, 2>();
    auto mask_T_a = mask_T.accessor<int64_t, 2>();
    auto mask_table_a = mask_table.accessor<int64_t, 2>();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0, 1);

    int64_t count = 0;
    for (int64_t i = 0; i < N_BLK_M; ++i) {
        for (int64_t j = 0; j < N_BLK_K; ++j) {
            bool drop = dis(gen) < p;
            uint64_t val = drop ? 1 : 0;
            mask_a[i][j / TMaskBits] |= (val << (j % TMaskBits));
            mask_T_a[j][i / TMaskBits] |= (val << (i % TMaskBits));

            if (!drop) {
                mask_table_a[count][0] = i;
                mask_table_a[count][1] = j;
                count++;
            }
        }
    }

    auto mask_cuda = mask.to(torch::kCUDA, true);
    auto mask_T_cuda = mask_T.to(torch::kCUDA, true);
    auto mask_table_cuda = mask_table.to(torch::kCUDA, true);

    return {mask_cuda, mask_T_cuda, mask_table_cuda, count};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> make_mask_from_existing(
    torch::Tensor m) {
    static constexpr int TMaskBits = 64;
    int64_t N_BLK_M = m.size(0);
    int64_t N_BLK_K = m.size(1);

    auto mask = torch::zeros({N_BLK_M, ct::ceil_div(N_BLK_K, TMaskBits)}, torch::kInt64);
    auto mask_T = torch::zeros({N_BLK_K, ct::ceil_div(N_BLK_M, TMaskBits)}, torch::kInt64);
    auto mask_table = torch::zeros({N_BLK_K * N_BLK_M, 2}, torch::kInt64);

    int64_t count = 0;
    for (int64_t i = 0; i < N_BLK_M; ++i) {
        for (int64_t j = 0; j < N_BLK_K; ++j) {
            bool val = m.index({i, j}).item<bool>();
            if (val) {
                int64_t mask_old = mask.index({i, j / TMaskBits}).item<int64_t>();
                int64_t mask_new = mask_old | (uint64_t(1) << (j % TMaskBits));
                int64_t mask_T_old = mask_T.index({j, i / TMaskBits}).item<int64_t>();
                int64_t mask_T_new = mask_T_old | (uint64_t(1) << (i % TMaskBits));

                mask.index_put_({i, j / TMaskBits}, mask_new);
                mask_T.index_put_({j, i / TMaskBits}, mask_T_new);
            }
            if (!val) {
                mask_table.index_put_({count, 0}, i);
                mask_table.index_put_({count, 1}, j);
                count++;
            }
        }
    }

    auto mask_cuda = mask.to(torch::kCUDA, true);
    auto mask_T_cuda = mask_T.to(torch::kCUDA, true);
    auto mask_table_cuda = mask_table.to(torch::kCUDA, true);

    return {mask_cuda, mask_T_cuda, mask_table_cuda, count};
}