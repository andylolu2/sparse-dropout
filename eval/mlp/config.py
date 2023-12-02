from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.seed = 0
    config.fabric = dict(
        accelerator="auto",
        precision="16-mixed",
    )

    config.model = dict(
        num_layers=3,
        hidden_dim=512,
        output_dim=10,
        variant="vanilla",
        p=0.1,
        block_size=(32, 64),
    )

    config.train = dict(
        lr=1e-3,
        max_epochs=200,
    )

    config.data = dict(
        train_batch_size=128,
        val_batch_size=128,
        train_size=1024,
        val_size=1024,
    )

    return config
