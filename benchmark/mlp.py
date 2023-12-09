import lightning as L
import torch
import triton
from absl import app

from eval.mlp.model import BasicNet

device = torch.device("cuda")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["hidden_dim"],
        x_vals=[i for i in range(256, 2048 + 1, 256)],
        line_arg="variant",
        line_vals=["none", "vanilla", "blockwise[triton]", "blockwise[naive]"],
        line_names=["None", "Vanilla", "Triton", "Naive"],
        styles=[("green", "-"), ("blue", "-"), ("cyan", "-"), ("red", "-")],
        ylabel="ms",
        plot_name="mlp-performance",
        args={"batch_size": 512, "block_size": (64, 64), "p": 0.9},
    )
)
def benchmark(hidden_dim, variant, batch_size, block_size, p):
    x = torch.randn(batch_size, 1, 28, 28, device=device, dtype=torch.float16)
    model = BasicNet(
        x,
        num_layers=3,
        hidden_dim=hidden_dim,
        output_dim=10,
        variant=variant,
        p=p,
        block_size=block_size,
    ).to(device, torch.float16)

    def f():
        loss = model(x).sum()
        loss.backward()

    ms, min_ms, max_ms = triton.testing.do_bench(f, quantiles=[0.5, 0.05, 0.95])

    def perf(ms):
        return ms
        # return 3 * 2 * M * N * K * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    L.seed_everything(0)
    benchmark.run(save_path="./logs", print_data=True)
