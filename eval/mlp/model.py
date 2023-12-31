import torch
from torch import nn

from flash_dropout.layers import DropoutMM


class BasicNet(nn.Module):
    def __init__(
        self,
        sample: torch.Tensor,
        num_layers: int,
        hidden_dim: int,
        output_dim: int,
        variant: str = "vanilla",
        p: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        assert num_layers >= 0

        self.in_proj = nn.Linear(sample[0].numel(), hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        self.layers = nn.ModuleList(
            [
                DropoutMM(hidden_dim, hidden_dim, p, variant, **kwargs)
                for _ in range(num_layers)
            ]
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = x.view(-1, x.shape[-1])
        x = self.in_proj(x)
        x = self.act(x)

        for layer in self.layers:
            x = layer(x)
            x = self.act(x)

        x = self.out_proj(x)
        return x
