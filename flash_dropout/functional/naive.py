import torch
import torch.nn.functional as F
import torch.types

from flash_dropout.types import size

from .utils import blockwise_dropout_mask, structured_dropout_mask


def blockwise_dropout(
    input: torch.Tensor, mask: torch.Tensor, block_size: size, p: float
):
    assert 0 <= p < 1, "Dropout probability must be in [0, 1)"
    M, K = input.shape
    BLK_M, BLK_K = block_size

    input_blocks = input.view(M // BLK_M, BLK_M, K // BLK_K, BLK_K).transpose(1, 2)
    input_blocks[mask] = 0
    input_blocks[~mask] /= 1 - p

    # mask = torch.repeat_interleave(mask, block_size[0], dim=0)
    # mask = torch.repeat_interleave(mask, block_size[1], dim=1)
    # mask = mask[: input.shape[0], : input.shape[1]]

    # x = input.clone()
    # x[mask] = 0
    # x[~mask] /= 1 - p

    return input


def structured_blockwise_dropout_matmul(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: size,
    p: float,
    training: bool = True,
):
    if training:
        mask = structured_dropout_mask(input, block_size, p).to(input.device)
        input = blockwise_dropout(input, mask, block_size, p)
    return input @ weight


def blockwise_dropout_matmul(
    input: torch.Tensor, weight: torch.Tensor, block_size: size, p: float
):
    mask = blockwise_dropout_mask(input, block_size, p)
    input = blockwise_dropout(input, mask, block_size, p)
    return F.linear(input, weight)
