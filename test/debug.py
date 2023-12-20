import lightning as L
import torch

from flash_dropout.cuda.binding import FlashDropoutCUDA

# from flash_dropout.functional.blockwise_dropout_matmul_triton import (
#     blockwise_dropout_matmul,
# )
from flash_dropout.functional.naive import (
    blockwise_dropout,
    blockwise_dropout_mask,
    blockwise_dropout_matmul,
)

L.seed_everything(0)
torch.set_printoptions(sci_mode=False, edgeitems=5, linewidth=200)

M, N, K = 512, 512, 512
block_size = (64, 32)
p = 0.5


def ref(A: torch.Tensor, B: torch.Tensor, mask, dC):
    A_ = A.clone().requires_grad_(True)
    B_ = B.clone().requires_grad_(True)
    C = blockwise_dropout(A_, mask, block_size, p) @ B_.T
    C.backward(dC)
    return C, A_.grad, B_.grad


def cuda_impl(A, B, mask, dC):
    A_ = A.clone()
    B_ = B.clone()
    ext = FlashDropoutCUDA(*block_size)
    C, _, mask_T, mask_table, count = ext.forward_test(A_, B_, mask.to("cpu"), p)
    dB, *_ = ext.backward(dC, A_, B_, mask_T, mask_table, p, count)
    return C, torch.zeros_like(A_), dB


A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(N, K, device="cuda", dtype=torch.float16)
dC = torch.randn(M, N, device="cuda", dtype=torch.float16)

mask = blockwise_dropout_mask(A, block_size, p)
print(f"{mask=}")


C_ref, dA_ref, dB_ref = ref(A, B, mask, dC)
C_cuda, dA_cuda, dB_cuda = cuda_impl(A, B, mask, dC)

# C = blockwise_dropout_matmul(A, B, block_size, 0.5)
# C.backward(torch.zeros_like(C))


def report(x_ref, x_cuda, name):
    max_abs_err = torch.max(torch.abs(x_cuda - x_ref))
    max_rel_err = torch.max(torch.abs((x_cuda - x_ref) / x_ref))

    print(f"{name}:")
    print("Reference:")
    print(x_ref)
    print("CUDA:")
    print(x_cuda)
    print(f"{max_abs_err=}")
    print(f"{max_rel_err=}")
    print()


report(C_ref, C_cuda, "C")
# report(dA_ref, dA_cuda, "dA")
report(dB_ref, dB_cuda, "dB")
