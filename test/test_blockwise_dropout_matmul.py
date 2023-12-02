from itertools import product

import lightning as L
import pytest
import torch

from flash_dropout.functional.blockwise_dropout_matmul_triton import (
    blockwise_dropout_matmul,
)
from flash_dropout.functional.naive import (
    blockwise_dropout_matmul as naive_blockwise_dropout_matmul,
)

block_sizes = [(16, 16), (64, 64), (32, 64)]
mnks = [
    (128, 128, 128),
    (32, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (1241, 511, 85),
]
ps = [0.0, 0.1, 0.5, 0.9, 0.99]


@pytest.mark.parametrize("mnk, block_size, p", list(product(mnks, block_sizes, ps)))
def test_blockwise_dropout_matmul(
    mnk: tuple[int, int, int],
    block_size: tuple[int, int],
    p: float,
):
    def run(f):
        L.seed_everything(0)
        M, N, K = mnk
        # Run in fp32 because the naive implementation is numerically unstable.
        A = torch.randn((M, K), device="cuda", dtype=torch.float32, requires_grad=True)
        B = torch.randn((K, N), device="cuda", dtype=torch.float32, requires_grad=True)
        grad = torch.randn((M, N), device="cuda", dtype=torch.float32)
        C = f(A, B, block_size, p)
        C.backward(grad)

        return A.grad, B.grad, C

    dA, dB, C_naive = run(naive_blockwise_dropout_matmul)
    dA_triton, dB_triton, C_triton = run(blockwise_dropout_matmul)

    torch.testing.assert_close(C_naive, C_triton, atol=0.2, rtol=0.01)
    torch.testing.assert_close(dB, dB_triton, atol=0.2, rtol=0.01)
    torch.testing.assert_close(dA, dA_triton, atol=0.2, rtol=0.01)
