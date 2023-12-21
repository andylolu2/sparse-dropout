import torch

from flash_dropout.cuda.binding import FlashDropoutCUDA
from flash_dropout.types import size


class BlockwiseDropoutMatmulCUDA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        block_size: size,
        p: float,
    ):
        assert 0 <= p < 1, "Dropout probability must be in [0, 1)"

        BLK_M, BLK_K = block_size
        impl = FlashDropoutCUDA(BLK_M, BLK_K)
        C, mask, mask_T, mask_table, count = impl.forward(input, weight, p)

        ctx.save_for_backward(input, weight, mask_T, mask_table)
        ctx.block_size = block_size
        ctx.count = count
        ctx.p = p

        return C

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ):
        input, weight, mask_T, mask_table = ctx.saved_tensors
        BLK_M, BLK_K = ctx.block_size
        count = ctx.count
        p = ctx.p

        impl = FlashDropoutCUDA(BLK_M, BLK_K)
        grad_input, grad_weight = impl.backward(
            grad_output, input, weight, mask_T, mask_table, p, count
        )

        return grad_input, grad_weight, None, None


def blockwise_dropout_matmul(
    input: torch.Tensor, weight: torch.Tensor, block_size: size, p: float
) -> torch.Tensor:
    return BlockwiseDropoutMatmulCUDA.apply(input, weight, block_size, p)  # type: ignore
