import torch

from flash_dropout.cuda.binding import FlashDropoutCUDA
from flash_dropout.types import size


class BlockwiseDropoutMatmulCUDA(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        block_size: size,
        p: float,
    ):
        assert block_size == (128, 128), "Only support block size (128, 128)"
        assert 0 <= p < 1, "Dropout probability must be in [0, 1)"

        # BLK_M, BLK_K = block_size
        impl = FlashDropoutCUDA(
            BLK_MNK_GROUP_0=(128, 128, 64, 5),
            BLK_MNK_GROUP_1=(128, 64, 128, 5),
            BLK_MNK_GROUP_2=(64, 128, 128, 5),
        )
        C, mask, mask_T, mask_table, count = impl.forward(input, weight, p)

        ctx.save_for_backward(input, weight, mask_T, mask_table)
        # ctx.block_size = block_size
        ctx.count = count  # type: ignore
        ctx.p = p  # type: ignore

        return C

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ):
        input, weight, mask_T, mask_table = ctx.saved_tensors  # type: ignore
        # BLK_M, BLK_K = ctx.block_size
        count = ctx.count  # type: ignore
        p = ctx.p  # type: ignore

        impl = FlashDropoutCUDA(
            BLK_MNK_GROUP_0=(128, 128, 64, 5),
            BLK_MNK_GROUP_1=(128, 64, 128, 5),
            BLK_MNK_GROUP_2=(64, 128, 128, 5),
        )
        grad_input = impl.backward_dA(grad_output, weight, mask_table, p, count)
        grad_weight = impl.backward_dB(grad_output, input, mask_T, p)

        return grad_input, grad_weight, None, None


def blockwise_dropout_matmul(
    input: torch.Tensor, weight: torch.Tensor, block_size: size, p: float
) -> torch.Tensor:
    return BlockwiseDropoutMatmulCUDA.apply(input, weight, block_size, p)  # type: ignore
