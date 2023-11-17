import torch
import torch.nn.functional as F
import torch.types

from structured_dropout.types import size

from .utils import dropout_mask


def blockwise_dropout(
    input: torch.Tensor, mask: torch.Tensor, block_size: size, p: float
):
    mask = torch.repeat_interleave(mask, block_size[0], dim=0)
    mask = torch.repeat_interleave(mask, block_size[1], dim=1)

    x = input.clone()
    x[mask] = 0
    x[~mask] *= 1 / (1 - p)

    return x


def blockwise_dropout_matmul(
    input: torch.Tensor,
    block_size: size,
    p: float,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    mask = dropout_mask(input, block_size, p)
    mask = torch.from_numpy(mask).to(input.device)
    input = blockwise_dropout(input, mask, block_size, p)
    return F.linear(input, weight, bias)
