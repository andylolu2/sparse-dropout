import math

import numpy as np
import torch
import torch.types


def blockwise_dropout(
    input: torch.Tensor,
    p: float = 0.5,
    block_size: torch.types._size = (16, 16),
    training: bool = False,
    inplace: bool = False,
):
    if training:
        return input
    else:
        N, K = input.shape
        BLK_N, BLK_K = block_size
        N_BLK_N, N_BLK_K = N // BLK_N, K // BLK_K

        x = input.clone() if inplace else input

        # Iterate over rows
        for i in range(N_BLK_N):
            # Drop roughly the same number of blocks per row of x
            for j in np.random.choice(N_BLK_K, math.ceil(N_BLK_K * p), replace=False):
                n = i * BLK_N
                k = j * BLK_K
                x[n : n + BLK_N, k : k + BLK_K] = 0

        return x


def blockwise_dropout_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    block_size: torch.types._size = (16, 16),
    p: float = 0.5,
    training: bool = False,
    inplace: bool = False,
):
    x = blockwise_dropout(x, p, block_size, training, inplace)
    return x @ weight


if __name__ == "__main__":
    N, M, K = 11, 12, 14
    input = torch.randint(10, (N, K))
    w = torch.randint(-10, 10, (K, M))

    y = blockwise_dropout_mm(input, w, block_size=(4, 4), p=0.2)

    print(input)
    print(y)

    # print(y)
