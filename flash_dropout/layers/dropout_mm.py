import math

import torch
import torch.types
from torch import nn

import flash_dropout.functional as F


class DropoutMM(nn.Module):
    __constants__ = ["in_features", "out_features"]
    p: float
    variant: str

    def __init__(
        self,
        in_features: int,
        out_features: int,
        p: float = 0.1,
        variant: str = "vanilla",
        **kwargs,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        self.variant = variant
        self.kwargs = kwargs

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return nn.functional.linear(input, self.weight)

        match self.variant:
            case "none":
                return nn.functional.linear(input, self.weight)
            case "vanilla":
                return F.vanilla_dropout_matmul(input, self.weight.T, self.p)
            case "blockwise[naive]":
                return F.naive_blockwise_dropout_matmul(
                    input, self.weight.T, self.kwargs["block_size"], self.p
                )
            case "blockwise[triton]":
                return F.triton_blockwise_dropout_matmul(
                    input, self.weight.T, self.kwargs["block_size"], self.p
                )
            case "blockwise[cuda]":
                return F.cuda_blockwise_dropout_matmul(
                    input, self.weight, self.kwargs["block_size"], self.p
                )
            case _:
                raise ValueError(f"Unknown variant {self.variant}")
