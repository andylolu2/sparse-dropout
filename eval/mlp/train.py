import lightning as L
import torch
import torch.nn.functional as F
from absl import app
from lightning.pytorch.callbacks import EarlyStopping
from ml_collections import config_flags
from torch.optim import Adam

from eval.mlp.dataset import MNISTDataModule
from eval.mlp.model import BasicNet
from eval.utils import Metrics

_CONFIG = config_flags.DEFINE_config_file("config", short_name="c")


def main(_):
    config = _CONFIG.value
    L.seed_everything(config.seed)

    fabric = L.Fabric(**config.fabric)
    fabric.launch()

    # Instantiate objects
    dm = MNISTDataModule(**config.data)
    dm.prepare_data()
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    model = BasicNet(dm.train_samlpe[0], **config.model)
    optimizer = Adam(model.parameters(), lr=config.train.lr, eps=1e-4)

    # Setup objects
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # Train
    metrics = Metrics()
    stopper = EarlyStopping(monitor="val_loss", patience=10)

    for epoch in range(config.train.max_epochs):
        model.train()
        for x, label in train_loader:
            optimizer.zero_grad()
            preds = model(x)
            losses = F.cross_entropy(preds, label, reduction="none")
            loss = losses.mean()
            fabric.backward(loss)
            optimizer.step()

            metrics.log(loss=losses.detach())

        model.eval()
        with torch.no_grad():
            for x, label in val_loader:
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


if __name__ == "__main__":
    app.run(main)
