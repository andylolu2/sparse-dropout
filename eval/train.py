import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn
from torch.optim import Adam

from eval.dataset import MNISTDataModule
from eval.utils import Metrics
from flash_dropout.layers import DropoutMM


class BasicNet(nn.Module):
    def __init__(
        self,
        sample: torch.Tensor,
        hidden_dim: int,
        n_classes: int = 10,
        variant: str = "vanilla",
        p: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dropout_kwargs = {"block_size": (32, 64)} if "blockwise" in variant else {}

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sample[0].numel(), hidden_dim),
            nn.ReLU(),
            DropoutMM(hidden_dim, hidden_dim, p, variant, **self.dropout_kwargs),
            nn.ReLU(),
            DropoutMM(hidden_dim, n_classes, p, variant, **self.dropout_kwargs),
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


if __name__ == "__main__":
    batch_size = 128
    hidden_dim = 512
    lr = 3e-4
    epochs = 200
    p = 0.6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant = "vanilla"

    L.seed_everything(0)

    dm = MNISTDataModule(batch_size, train_size=1024, val_size=1024)
    dm.prepare_data()

    dm.setup("fit")

    model = BasicNet(dm.train_samlpe[0], hidden_dim, variant=variant, p=p).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    metrics = Metrics()

    stopper = EarlyStopping(monitor="val_loss", patience=10)

    for epoch in range(epochs):
        model.train()
        for batch_id, (x, label) in enumerate(dm.train_dataloader()):
            x, label = dm.transfer_batch_to_device((x, label), device, batch_id)
            optimizer.zero_grad()
            preds = model(x)
            losses = F.cross_entropy(preds, label, reduction="none")
            loss = losses.mean()
            loss.backward()
            optimizer.step()

            metrics.log(loss=losses.detach())

        model.eval()
        with torch.no_grad():
            for batch_id, (x, label) in enumerate(dm.val_dataloader()):
                x, label = dm.transfer_batch_to_device((x, label), device, batch_id)
                preds = model(x)
                losses = F.cross_entropy(preds, label, reduction="none")
                metrics.log(val_loss=losses)

        logs = metrics.asdict()
        train_loss = torch.concatenate(logs["loss"]).mean()
        val_loss = torch.concatenate(logs["val_loss"]).mean()
        print(
            f"Epoch {epoch + 1}: train/loss={train_loss.item():.4f}, val/loss={val_loss.item():.4f}"
        )
        metrics.clear()

        should_stop, reason = stopper._evaluate_stopping_criteria(val_loss)
        if should_stop:
            print(f"Early stopping: {reason}")
            break

    dm.teardown("fit")
