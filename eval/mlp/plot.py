import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from absl import app
from wandb.apis.public import Run

import wandb


def load_runs(entity: str, project: str, run_ids: list[int]) -> list[Run]:
    api = wandb.Api()

    runs = api.runs(
        path=f"{entity}/{project}",
        filters={
            "display_name": {
                "$regex": rf"^[a-zA-Z]+-[a-zA-Z]+-({'|'.join(map(str, run_ids))})$"
            }
        },
    )

    return runs


def main(_):
    import miscellaneous.plot_style

    wandb.login()
    run_ids = list(range(7, 25)) + list(range(31, 49))
    runs = load_runs("andylolu2", "flash-dropout-mlp", run_ids)

    metrics = {
        "train/acc": ("Train accuracy", "max"),
        "train/loss": ("Train loss", "min"),
        "val/acc": ("Accuracy", "max"),
        "val/loss": ("Loss", "min"),
    }

    data = []
    for run in runs:
        logs = list(run.scan_history(page_size=10000))
        logs_df = pd.DataFrame(logs)

        item = {
            "p": run.config["model"]["dropout"]["p"],
            "Variant": run.config["model"]["dropout"]["variant"],
        }
        for metric, (_, mode) in metrics.items():
            agg = np.nanmax if mode == "max" else np.nanmin
            item[metric] = agg(logs_df[metric])

        data.append(item)

    df = pd.DataFrame(data)
    print(df.sort_values(["Variant", "p"]))

    for metric, (name, mode) in metrics.items():
        print("\n", f"--- {name} ({metric}) ---")

        def sem(x):
            return x.sem(ddof=0)

        stats = df.groupby(["Variant", "p"])[metric].agg(["mean", sem])
        stats["interval"] = 1.96 * stats["sem"]
        stats = stats.sort_values(["mean"])
        print(stats)

        # fig, ax = plt.subplots(figsize=(5, 4))
        # sns.lineplot(
        #     data=df,
        #     x="p",
        #     y=metric,
        #     hue="Variant",
        #     marker="o",
        #     markersize=4,
        #     ax=ax,
        # )
        # ax.set(
        #     xlim=(-0.02, None),
        #     xlabel="$p$",
        #     ylabel=name,
        # )

        # fig.tight_layout()
        # fig.savefig(f"./logs/mlp-{metric.replace('/', '-')}.png", dpi=300)


if __name__ == "__main__":
    app.run(main)
