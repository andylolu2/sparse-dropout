import math
import time

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.types

import flash_dropout.functional as F
from flash_dropout.cuda.binding import FlashDropoutCUDA

block_size = (128, 128)


def f_naive(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor, p: float):
    C = F.naive_blockwise_dropout_matmul(A, B, block_size, p)
    yield C

    C.backward(dC)
    yield A.grad, B.grad


def f_dense(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor):
    C = torch.nn.functional.linear(A, B)
    yield C

    C.backward(dC)
    yield A.grad, B.grad


def f_baseline(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor, p: float):
    C = F.vanilla_dropout_matmul(A, B, p)
    yield C

    C.backward(dC)
    yield A.grad, B.grad


def f_cuda(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor, p: float):
    impl = FlashDropoutCUDA(
        BLK_MNK_GROUP_0=(128, 128, 64, 5),
        BLK_MNK_GROUP_1=(128, 64, 128, 5),
        BLK_MNK_GROUP_2=(64, 128, 128, 5),
        # BLK_MNK_GROUP_0=(64, 64, 64, 5),
        # BLK_MNK_GROUP_1=(64, 64, 64, 5),
        # BLK_MNK_GROUP_2=(64, 64, 64, 5),
    )

    C, mask, mask_T, mask_table, count = impl.forward(A, B, p)
    yield C

    dA = impl.backward_dA(dC, B, mask_table, p, count)
    dB = impl.backward_dB(dC, A, mask_T, p)
    yield dA, dB


def f_triton(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor, p: float):
    C = F.triton_blockwise_dropout_matmul(A, B, block_size, p)
    yield C

    C.backward(dC)
    yield A.grad, B.grad


def do_bench_detailed(fn, warmup=0.5, rep=0.2):
    n_breakpoints = 0
    for _ in fn():
        n_breakpoints += 1
    torch.cuda.synchronize()

    cache = torch.empty(int(4e6 // 8), dtype=torch.int64, device="cuda")

    events = []

    start_time = time.time()
    while time.time() - start_time < warmup:
        cache.zero_()
        for _ in fn():
            pass
        torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    while time.time() - start_time < rep:
        breakpoint_events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(n_breakpoints)
        ]
        events.append(breakpoint_events)

        cache.zero_()
        f = iter(fn())
        for start, end in breakpoint_events:
            start.record()
            next(f)
            end.record()
        torch.cuda.synchronize()

    # Record clocks
    torch.cuda.synchronize()
    breakpoint_times = []
    for breakpoint_events in events:
        times = []
        for start, end in breakpoint_events:
            times.append(start.elapsed_time(end))
        breakpoint_times.append(times)
    times = torch.tensor(breakpoint_times, dtype=torch.float32)
    return times.T


if __name__ == "__main__":
    import miscellaneous.plot_style

    L.seed_everything(0)

    # b = 4
    # s = 256
    # d = 512
    # M, N, K = b * s, d * 4, d
    # M, N, K = b * s, d, d * 4
    # M, N, K = b * s, d, 3 * d
    M, N, K = 1024, 1024, 1024

    A = torch.randn((M, K), device="cuda", dtype=torch.float16, requires_grad=True)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16, requires_grad=True)
    dC = torch.randn((M, N), device="cuda", dtype=torch.float16)

    breakpoint_names = ["forward", "backward"]

    dense_fs = {
        "Dense": lambda: f_dense(A, B, dC),
        "Dropout + Dense": lambda: f_baseline(A, B, dC, 0.5),
        "Block dropout + Dense": lambda: f_naive(A, B, dC, 0.5),
    }

    sparse_fs = {
        "SparseDrop": lambda p: f_cuda(A, B, dC, p),
        # "SparseDrop Triton": lambda p: f_triton(A, B, dC, p),
    }

    dense_data = {}
    for name, fn in dense_fs.items():
        timings = do_bench_detailed(fn, warmup=1, rep=1)
        dense_data[name] = {}
        for breakpoint_name, timing in zip(breakpoint_names, timings):
            avg = timing.mean().item()
            std_err = timing.std().item() / math.sqrt(timing.shape[0])
            dense_data[name][breakpoint_name] = {
                "avg": avg,
                "std": std_err,
                "flops": 2 * M * N * K / (avg / 1000),
            }
        full_timings = timings.sum(dim=0)
        avg = full_timings.mean().item()
        std_err = full_timings.std().item() / math.sqrt(full_timings.shape[0])
        dense_data[name]["full"] = {
            "avg": avg,
            "std": std_err,
            "flops": 6 * M * N * K / (avg / 1000),
        }

    data = []
    for name, fn in sparse_fs.items():
        for p in np.arange(0.0, 1.0, 0.05):
            timings = do_bench_detailed(lambda: fn(p))
            for breakpoint_name, timing in zip(breakpoint_names, timings):
                for t in timing:
                    data.append(
                        {
                            "name": name,
                            "breakpoint": breakpoint_name,
                            "time": t.item(),
                            "sparsity": p,
                            "flops": 2 * M * N * K * (1 - p) / (t.item() / 1000),
                        }
                    )
            full_timings = timings.sum(dim=0)
            for t in full_timings:
                data.append(
                    {
                        "name": name,
                        "breakpoint": "full",
                        "time": t.item(),
                        "sparsity": p,
                        "flops": 6 * M * N * K * (1 - p) / (t.item() / 1000),
                    }
                )
    df = pd.DataFrame(data)
    df.rename(columns={"name": "Method"}, inplace=True)

    for breakpoint_name in df["breakpoint"].unique():
        fig, ax = plt.subplots(figsize=(5, 4))

        sub_df = df[df["breakpoint"] == breakpoint_name]
        for i, (name, item) in enumerate(dense_data.items()):
            avg = item[breakpoint_name]["avg"]
            ax.axhline(avg, label=name, linestyle="--", color=f"C{i+1}")
            # lower = avg - 1.96 * item[breakpoint_name]["std"]
            # upper = avg + 1.96 * item[breakpoint_name]["std"]
            # ax.fill_between(
            #     [0, 1],
            #     [lower, lower],
            #     [upper, upper],
            #     color=f"C{i+1}",
            #     alpha=0.2,
            # )

        sns.lineplot(
            data=sub_df,
            x="sparsity",
            y="time",
            hue="Method",
            marker="o",
            markersize=4,
            errorbar=None,
            ax=ax,
        )
        ax.set(
            xlim=(-0.02, 1.02),
            ylabel="Time (ms)",
            xlabel="Sparsity",
        )
        fig.tight_layout()
        fig.savefig(
            f"./logs/matmul-detailed_m{M}n{N}k{K}_{breakpoint_name}.png", dpi=300
        )

        fig, ax = plt.subplots(figsize=(5, 4))

        x_intercept = dense_data["Dense"][breakpoint_name]["flops"]
        ax.plot(
            [0, 1],
            [x_intercept, 0],
            linestyle="--",
            color="C1",
            label="FLOPS for speed-up",
        )

        sns.lineplot(
            data=sub_df,
            x="sparsity",
            y="flops",
            hue="Method",
            marker="o",
            markersize=4,
            errorbar=None,
            ax=ax,
        )
        ax.set(
            xlim=(-0.02, 1.02),
            ylabel="FLOPS",
            xlabel="Sparsity",
        )
        fig.tight_layout()
        fig.savefig(
            f"./logs/matmul-detailed_m{M}n{N}k{K}_{breakpoint_name}_flops.png", dpi=300
        )
