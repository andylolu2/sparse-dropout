from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from structured_dropout.layers import BlockwiseDropout


class BasicNet(nn.Module):
    def __init__(
        self,
        sample: torch.Tensor,
        hidden_dim: int = 128,
        n_classes: int = 10,
        dropout_variant: str = "vanilla",
    ):
        super().__init__()

        match dropout_variant:
            case "vanilla":
                dropout = partial(nn.Dropout, p=0.1)
            case "blockwise":
                dropout = partial(BlockwiseDropout, p=0.1)
            case "none":
                dropout = nn.Identity
            case _:
                raise ValueError()

        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sample[0].numel(), hidden_dim),
            nn.ReLU(),
            dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


if __name__ == "__main__":
    batch_size = 128

    train_loader = DataLoader(
        MNIST("./data", download=True, train=True, transform=ToTensor()),
        batch_size=batch_size,
        shuffle=True,
    )

    model = BasicNet(next(iter(train_loader))[0], dropout_variant="blockwise")
    optimizer = Adam(model.parameters(), lr=3e-4)

    for epoch in range(5):
        for batch_id, (x, label) in enumerate(train_loader):
            optimizer.zero_grad()
            preds = model(x)
            loss = F.cross_entropy(preds, label)
            loss.backward()
            optimizer.step()

            if batch_id % 100 == 0:
                loss = loss.item()
                print(f"{epoch = } {batch_id = } {loss = }")
