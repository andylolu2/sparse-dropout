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
torch.set_printoptions(sci_mode=False, edgeitems=5, linewidth=5000)

M, N, K = 512, 512, 512
block_size = (64, 32)
p = 0.5


def ref(A: torch.Tensor, B: torch.Tensor, mask: torch.Tensor, dC: torch.Tensor):
    A = blockwise_dropout(A, mask, block_size, p)
    C = A @ B.T
    return C, blockwise_dropout(dC @ B, mask, block_size, p), dC.T @ A


def cuda_impl(A: torch.Tensor, B: torch.Tensor, mask: torch.Tensor, dC: torch.Tensor):
    ext = FlashDropoutCUDA(*block_size)
    C, _, mask_T, mask_table, count = ext.forward_test(A, B, mask.to("cpu"), p)
    dA, dB = ext.backward(dC, A, B, mask_T, mask_table, p, count)
    return C, dA, dB


A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(N, K, device="cuda", dtype=torch.float16)
dC = torch.randn(M, N, device="cuda", dtype=torch.float16)

mask = blockwise_dropout_mask(A, block_size, p)
print(f"{mask=}")


C_ref, dA_ref, dB_ref = ref(A, B, mask, dC)
C_cuda, dA_cuda, dB_cuda = cuda_impl(A, B, mask, dC)


def report(x_ref, x_cuda, name):
    abs_err = torch.abs(x_cuda - x_ref)
    rel_err = torch.abs((x_cuda - x_ref) / x_ref).nan_to_num()

    max_abs_err_idx = torch.argmax(abs_err)
    max_abs_err_abs = abs_err.flatten()[max_abs_err_idx].item()
    max_abs_err_rel = rel_err.flatten()[max_abs_err_idx].item()

    max_rel_err_idx = torch.argmax(rel_err)
    max_rel_err_abs = abs_err.flatten()[max_rel_err_idx].item()
    max_rel_err_rel = rel_err.flatten()[max_rel_err_idx].item()

    print(f"{name}:")
    print("Reference:")
    print(x_ref)
    print("CUDA:")
    print(x_cuda)
    print(f"{max_abs_err_abs=} {max_abs_err_rel=:.1%}")
    print(f"{max_rel_err_abs=} {max_rel_err_rel=:.1%}")
    print()


report(C_ref, C_cuda, "C")
report(dA_ref, dA_cuda, "dA")
report(dB_ref, dB_cuda, "dB")
