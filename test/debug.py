import lightning as L
import torch

from flash_dropout.cuda.binding import forward_test

# from flash_dropout.functional.blockwise_dropout_matmul_triton import (
#     blockwise_dropout_matmul,
# )
from flash_dropout.functional.naive import (
    blockwise_dropout,
    blockwise_dropout_mask,
    blockwise_dropout_matmul,
)

L.seed_everything(0)
torch.set_printoptions(sci_mode=False, edgeitems=5, linewidth=120)


def ref(A, B, mask):
    A = blockwise_dropout(A, mask, block_size, p=0)
    return A @ B.T


def cuda_impl(A, B, mask):
    C, *_ = forward_test(A, B, mask)
    return C


M, N, K = 512, 512, 512
block_size = (64, 64)
p = 0.5

A = torch.randn(M, K, device="cuda", dtype=torch.float16, requires_grad=True)
B = torch.randn(N, K, device="cuda", dtype=torch.float16, requires_grad=True)


mask = blockwise_dropout_mask(A, block_size, p)
print(f"{mask=}")

C_ref = ref(A, B, mask)
C_cuda = cuda_impl(A, B, mask)

# C = blockwise_dropout_matmul(A, B, block_size, 0.5)
# C.backward(torch.zeros_like(C))

print(f"{C_ref=}")
print(f"{C_cuda=}")

max_abs_err = torch.max(torch.abs(C_cuda - C_ref))
max_rel_err = torch.max(torch.abs((C_cuda - C_ref) / C_ref))
print(f"{max_abs_err=}")
print(f"{max_rel_err=}")
