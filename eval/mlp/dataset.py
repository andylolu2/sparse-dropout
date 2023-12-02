from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_batch_size: int,
        train_size: int = 55000,
        val_size: int = 5000,
        val_batch_size: int | None = None,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size or train_batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.data_dir = str(Path.home() / ".cache" / "torchvision" / "mnist")
        self.transform = transforms.ToTensor()

    @property
    def train_samlpe(self):
        return self.mnist_train[0]

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            indices = np.random.choice(
                len(mnist_full), self.train_size + self.val_size, replace=False
            ).tolist()
            self.mnist_train = Subset(mnist_full, indices[: self.train_size])
            self.mnist_val = Subset(mnist_full, indices[self.train_size :])
        elif stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )
        elif stage == "predict":
            self.mnist_predict = MNIST(
                self.data_dir, train=False, transform=self.transform
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, batch_size=self.train_batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.val_batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.val_batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.val_batch_size)
