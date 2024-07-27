#pragma once

#include <torch/extension.h>

#include <random>

torch::Tensor make_mask(int64_t m, int64_t n, int64_t block_size, float p) {
    assert(m % block_size == 0 && "M must be divisible by block_size");
    assert(n % block_size == 0 && "N must be divisible by block_size");

    torch::Tensor mask = torch::empty({m / block_size, n / block_size}, torch::kBool);

    auto mask_a = mask.accessor<bool, 2>();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 1);

    for (int64_t i = 0; i < mask.size(0); ++i) {
        for (int64_t j = 0; j < mask.size(1); ++j) {
            mask_a[i][j] = dis(gen) < p;
        }
    }

    auto mask_cuda = mask.to(torch::kCUDA, true);

    return mask_cuda;
}