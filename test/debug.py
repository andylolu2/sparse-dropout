import torch

from flash_dropout.functional.blockwise_dropout_matmul_triton import (
    blockwise_dropout_matmul,
)

A = torch.randn(128, 128, device="cuda", dtype=torch.float16, requires_grad=True)
B = torch.randn(128, 128, device="cuda", dtype=torch.float16, requires_grad=True)

C = blockwise_dropout_matmul(A, B, (64, 64), 0.5)
C.backward(torch.zeros_like(C))
