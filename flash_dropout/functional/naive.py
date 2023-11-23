import torch
import torch.nn.functional as F
import torch.types

from flash_dropout.types import size

from .utils import blockwise_dropout_mask, flash_dropout_mask


def blockwise_dropout(
    input: torch.Tensor, mask: torch.Tensor, block_size: size, p: float
):
    mask = torch.repeat_interleave(mask, block_size[0], dim=0)
    mask = torch.repeat_interleave(mask, block_size[1], dim=1)

    x = input.clone()
    x[mask] = 0
    x[~mask] /= 1 - p

    return x


def structured_blockwise_dropout_matmul(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: size,
    p: float,
):
    mask = flash_dropout_mask(input, block_size, p).to(input.device)
    input = blockwise_dropout(input, mask, block_size, p)
    return input @ weight


def blockwise_dropout_matmul(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: size,
    p: float,
):
    mask = blockwise_dropout_mask(input, block_size, p).to(input.device)
    input = blockwise_dropout(input, mask, block_size, p)
    return input @ weight
