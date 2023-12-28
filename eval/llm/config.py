from pathlib import Path

from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_config():
    config = ConfigDict()

    config.seed = 0
    config.fabric = dict(
        accelerator="auto",
        precision="16-mixed",
    )
    config.wandb = dict(
        project="flash-dropout",
        notes=placeholder(str),
        mode="online",
    )

    config.model = dict(
        context_length=256,
        vocab_size=256,
        n_layer=6,
        n_head=8,
        n_embed=512,
        batch_first=True,
        dropout=dict(
            p=0.0,
            variant="vanilla",
            block_size=(128, 128),
        ),
    )

    config.optimizer = dict(
        lr=1e-3,
        weight_decay=1e-1,
    )

    config.train = dict(
        max_epochs=200,
        early_stop_patience=10,
    )

    config.data = dict(
        length=config.model.get_ref("context_length"),
        batch_first=config.model.get_ref("batch_first"),
        vocab_size=config.model.get_ref("vocab_size"),
        cache_dir=Path("data", "wikitext"),
        train_batch_size=4,
        val_batch_size=32,
        train_size=512,
        val_size=512,
    )

    return config
