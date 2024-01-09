import torch
import torch.nn.functional as F
import torch.types

from flash_dropout.types import size

from .utils import blockwise_dropout_mask


def blockwise_dropout_matmul(
    input: torch.Tensor, weight: torch.Tensor, block_size: size, p: float
):
    """Experimental: This does not support backward pass."""
    M, K = input.shape
    BLK_M, BLK_K = block_size

    mask = blockwise_dropout_mask(input, block_size, p)
    smask = mask.to_sparse_csr()

    input_blocks = input.view(M // BLK_M, BLK_M, K // BLK_K, BLK_K).transpose(1, 2)
    input_bsr = torch.sparse_bsr_tensor(
        crow_indices=smask.crow_indices(),
        col_indices=smask.col_indices(),
        values=input_blocks[mask],
        size=input.shape,
        requires_grad=input.requires_grad,
    )

    return F.linear(input_bsr, weight)
