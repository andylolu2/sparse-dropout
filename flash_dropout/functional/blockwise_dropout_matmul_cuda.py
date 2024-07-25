import torch

from flash_dropout.cuda.binding_gemm import GEMM


class BlockwiseDropoutMatmulCUDA(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        block_size: int,
        p: float,
    ):
        assert 0 <= p < 1, "Dropout probability must be in [0, 1)"

        M, N, K = input.shape[0], weight.shape[0], input.shape[1]

        ext = GEMM()
        mask = torch.rand(M // block_size, K // block_size, device="cuda") < p

        # C (M N) = A (M K sparse) * B (N K)
        C = ext.gemm_dsd(input, weight, mask, block_size)

        ctx.save_for_backward(input, weight, mask)
        ctx.block_size = block_size  # type: ignore
        ctx.p = p  # type: ignore

        return C

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ):
        input, weight, mask = ctx.saved_tensors  # type: ignore
        block_size = ctx.block_size  # type: ignore
        p = ctx.p  # type: ignore

        ext = GEMM()
        # dA (M K sparse) = dC (M N) * B.T (K N)
        grad_input = ext.gemm_sdd(grad_output, weight.T, mask, block_size)
        # dB (N K) = dC.T (N M) * A.T (K M sparse)
        #          = (A.T (K M sparse) * dC.T (N M)).T
        grad_weight = ext.gemm_dsd(input.T, grad_output.T, mask.T, block_size).T
        # grad_input = impl.backward_dA(grad_output, weight, mask_table, p, count)
        # grad_weight = impl.backward_dB(grad_output, input, mask_T, p)

        return grad_input, grad_weight, None, None


def blockwise_dropout_matmul(
    input: torch.Tensor, weight: torch.Tensor, block_size: int, p: float
) -> torch.Tensor:
    return BlockwiseDropoutMatmulCUDA.apply(input, weight, block_size, p)  # type: ignore
