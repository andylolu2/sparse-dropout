import torch
import torch.types
from torch import nn

import flash_dropout.functional as F


class BlockwiseDropout(nn.Module):
    __constants__ = ["p", "inplace"]
    p: float
    inplace: bool
    block_size: torch.types._size

    def __init__(
        self,
        p: float = 0.5,
        block_size: torch.types._size = (16, 16),
        inplace: bool = False,
    ) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                f"dropout probability has to be between 0 and 1, but got {p}"
            )

        self.p = p
        self.block_size = block_size
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.blockwise_dropout(
            input, self.p, self.block_size, self.training, self.inplace
        )
