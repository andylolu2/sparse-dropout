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
        assert num_layers >= 1

        self.layers = nn.ModuleList([nn.Flatten()])
        dims = [sample[0].numel(), *([hidden_dim] * (num_layers - 1)), output_dim]
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-2], dims[1:-1])):
            if i == 0:
                self.layers.append(nn.Linear(in_dim, out_dim))
            else:
                self.layers.append(DropoutMM(in_dim, out_dim, p, variant, **kwargs))
            self.layers.append(nn.ReLU())
        self.layers += [nn.Linear(dims[-2], dims[-1])]

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
