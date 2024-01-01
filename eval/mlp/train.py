import lightning as L
import torch
import torch.nn.functional as F
from absl import app
from lightning.pytorch.callbacks import EarlyStopping
from ml_collections import config_flags
from torch.optim import Adam

import wandb
from eval.mlp.dataset import MNISTDataModule
from eval.mlp.model import BasicNet
from eval.utils import CudaTimer, Metrics

_CONFIG = config_flags.DEFINE_config_file("config", short_name="c")


def main(_):
    config = _CONFIG.value

    wandb.login()
    wandb.init(**config.wandb, config=config.to_dict())

    L.seed_everything(config.seed)

    fabric = L.Fabric(**config.fabric)
    fabric.launch()

    # Instantiate objects
    dm = MNISTDataModule(**config.data)
    dm.prepare_data()
    dm.setup("fit")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    model = BasicNet(dm.train_samlpe[0], **config.model)
    model = fabric.setup_module(model)

    optimizer = Adam(model.parameters(), **config.optimizer)
    optimizer = fabric.setup_optimizers(optimizer)

    # Train
    metrics = Metrics()
    stopper = EarlyStopping(
        monitor="val_loss", patience=config.train.early_stop_patience
    )

    step = 0
    for epoch in range(config.train.max_epochs):
        model.train()
        for x, label in train_loader:
            optimizer.zero_grad()
            with CudaTimer() as fwd_timer:
                preds = model(x)
                losses = F.cross_entropy(preds, label, reduction="none")
                loss = losses.mean()
            with CudaTimer() as bwd_timer:
                fabric.backward(loss)
            optimizer.step()

            metrics.log(
                loss=losses.detach(),
                fwd_time=fwd_timer.elapsed(),
                bwd_time=bwd_timer.elapsed(),
            )
            step += 1

        model.eval()
        with torch.no_grad():
            for x, label in val_loader:
                preds = model(x)
                losses = F.cross_entropy(preds, label, reduction="none")
                metrics.log(val_loss=losses)

        logs = metrics.asdict()
        metrics.clear()
        train_loss = torch.concatenate(logs["loss"]).mean()
        val_loss = torch.concatenate(logs["val_loss"]).mean()
        fwd_time = torch.tensor(logs["fwd_time"]).mean()
        bwd_time = torch.tensor(logs["bwd_time"]).mean()
        wandb.log(
            {
                "train/loss": train_loss.item(),
                "val/loss": val_loss.item(),
                "fwd_time": fwd_time.item(),
                "bwd_time": bwd_time.item(),
                "step": step,
                "epoch": epoch,
            },
        )

        should_stop, reason = stopper._evaluate_stopping_criteria(val_loss)
        if should_stop:
            print(f"Early stopping: {reason}")
            break

    dm.teardown("fit")


if __name__ == "__main__":
    app.run(main)
