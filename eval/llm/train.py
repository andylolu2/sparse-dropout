import lightning as L
import torch
from absl import app
from lightning.pytorch.callbacks import EarlyStopping
from ml_collections import config_flags

import wandb
from eval.llm.dataset import WikitextDataModule
from eval.llm.model import GPT
from eval.utils import CudaTimer, global_metrics

_CONFIG = config_flags.DEFINE_config_file("config", short_name="c")


def main(_):
    config = _CONFIG.value
    print(config)

    wandb.login()
    wandb.init(**config.wandb, config=config.to_dict())

    L.seed_everything(config.seed)

    fabric = L.Fabric(**config.fabric)
    fabric.launch()

    # Instantiate objects
    dm = WikitextDataModule(**config.data)
    dm.prepare_data()
    dm.setup("fit")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    model = GPT(**config.model)
    model = fabric.setup_module(model)

    optimizer = model.configure_optimizers(**config.optimizer)
    optimizer = fabric.setup_optimizers(optimizer)

    # Train
    stopper = EarlyStopping(
        monitor="val_loss", patience=config.train.early_stop_patience
    )

    step = 0
    for epoch in range(config.train.max_epochs):
        model.train()
        for x in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with CudaTimer() as fwd_timer:
                logits, loss = model(x)
            with CudaTimer() as bwd_timer:
                fabric.backward(loss)
            optimizer.step()

            global_metrics.log(
                loss=loss.detach(),
                fwd_time=fwd_timer.elapsed(),
                bwd_time=bwd_timer.elapsed(),
            )
            step += 1

        model.eval()
        with torch.no_grad():
            for x in val_loader:
                logits, loss = model(x)
                global_metrics.log(val_loss=loss)

        logs = global_metrics.asdict()
        global_metrics.clear()
        train_loss = torch.tensor(logs["loss"]).mean()
        val_loss = torch.tensor(logs["val_loss"]).mean()
        fwd_time = torch.tensor(logs["fwd_time"]).mean()
        bwd_time = torch.tensor(logs["bwd_time"]).mean()
        mfu = model.estimate_mfu(
            dm.train_batch_size, (fwd_time.item() + bwd_time.item()) / 1000
        )
        wandb.log(
            {
                "train/loss": train_loss.item(),
                "val/loss": val_loss.item(),
                "mfu": mfu,
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
