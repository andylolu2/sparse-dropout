from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.seed = 0
    config.fabric = dict(
        accelerator="auto",
        precision="16-true",
    )

    config.model = dict(
        num_layers=5,
        hidden_dim=512,
        output_dim=10,
        variant="blockwise[cuda]",
        p=0.0,
        block_size=(128, 128),
    )

    config.train = dict(
        lr=1e-3,
        max_epochs=200,
    )

    config.data = dict(
        train_batch_size=512,
        val_batch_size=512,
        train_size=4096,
        val_size=8192,
    )

    return config
