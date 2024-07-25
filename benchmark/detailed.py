import time
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.types

import flash_dropout.functional as F
from flash_dropout.cuda.binding_gemm import GEMM
from flash_dropout.functional.blockwise_dropout_matmul_cuda import (
    blockwise_dropout_matmul as blockwise_dropout_matmul_cuda,
)
from flash_dropout.functional.utils import blockwise_dropout_mask
from flash_dropout.triton.dsd_matmul import blockwise_dsd_matmul
from flash_dropout.triton.sdd_matmul import blockwise_sdd_matmul

block_size = 128
CACHE = {}


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


def f_dense_cuda(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor):
    ext = GEMM()
    C = ext.gemm(A, B)
    yield C

    dA = ext.gemm(dC, B.T)
    dB = ext.gemm(A.T, dC.T).T
    yield dA, dB


def f_cuda(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor, p: float):
    C = blockwise_dropout_matmul_cuda(A, B, block_size, p)
    yield C

    C.backward(dC)
    yield A.grad, B.grad


def f_triton_fixed_mask(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor, p: float):
    """
    Args:
        A: M K
        B: N K
        dC: M N
    """
    if f"triton_{p}" not in CACHE:
        mask = blockwise_dropout_mask(A, block_size, p)
        mask_T = mask.T
        table = torch.nonzero(~mask, as_tuple=False)
        CACHE[f"triton_{p}"] = (mask, mask_T, table)
    else:
        mask, mask_T, table = CACHE[f"triton_{p}"]
    C = blockwise_dsd_matmul(A, mask, B.T, block_size, 1 / (1 - p))
    yield C

    dA = blockwise_sdd_matmul(dC, B, table, block_size, 1 / (1 - p))
    dB = blockwise_dsd_matmul(A.T, mask_T, dC, block_size, 1 / (1 - p)).T
    yield dA, dB


def f_triton(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor, p: float):
    """
    Args:
        A: M K
        B: N K
        dC: M N
    """
    mask = blockwise_dropout_mask(A, block_size, p)
    C = blockwise_dsd_matmul(A, mask, B.T, block_size, 1 / (1 - p))
    yield C

    mask_T = mask.T
    table = torch.nonzero(~mask, as_tuple=False)
    dA = blockwise_sdd_matmul(dC, B, table, block_size, 1 / (1 - p))
    dB = blockwise_dsd_matmul(A.T, mask_T, dC, block_size, 1 / (1 - p)).T
    yield dA, dB


def do_bench_detailed(fn, warmup: int = 10, rep: float = 0.5):
    n_breakpoints = 0
    for _ in fn():
        n_breakpoints += 1

    cache = torch.empty(4 * 1024**2, dtype=torch.int8, device="cuda")

    for _ in range(warmup):
        cache.zero_()
        for _ in fn():
            pass

    # Benchmark
    events = []
    start_time = time.time()
    while time.time() - start_time < rep:
        breakpoint_events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(n_breakpoints)
        ]
        events.append(breakpoint_events)

        # cache.zero_()
        f = iter(fn())
        for start, end in breakpoint_events:
            start.record()  # type: ignore
            next(f)
            end.record()  # type: ignore
        torch.cuda.synchronize()

    # Record clocks
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
    # M, N, K = 1024, 1024, 1024
    # M, N, K = 2048, 2048, 2048
    M, N, K = 4096, 4096, 4096

    A = torch.randn((M, K), device="cuda", dtype=torch.float16, requires_grad=True)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16, requires_grad=True)
    dC = torch.randn((M, N), device="cuda", dtype=torch.float16)

    breakpoint_names = ["forward", "backward"]

    dense_fs = {
        "Dense": lambda: f_dense(A, B, dC),
        "Dense CUDA": lambda: f_dense_cuda(A, B, dC),
        "Dropout + Dense": lambda: f_baseline(A, B, dC, 0.5),
        "Block dropout + Dense": lambda: f_naive(A, B, dC, 0.5),
    }

    sparse_fs = {
        "CUDA": lambda p: f_cuda(A, B, dC, p),
        # "Triton": lambda p: f_triton(A, B, dC, p),
        # "Triton (fixed mask)": lambda p: f_triton_fixed_mask(A, B, dC, p),
    }

    dense_data = {}
    for name, fn in dense_fs.items():
        timings = do_bench_detailed(fn)
        dense_data[name] = {}
        for breakpoint_name, timing in zip(breakpoint_names, timings):
            median = timing.median().item()
            dense_data[name][breakpoint_name] = {
                "median": median,
                "flops": 2 * M * N * K / (median / 1000),
            }
        full_timings = timings.sum(dim=0)
        median = full_timings.median().item()
        dense_data[name]["full"] = {
            "median": median,
            "flops": 6 * M * N * K / (median / 1000),
        }

    data = []
    for name, sparse_fn in sparse_fs.items():
        for p in np.arange(0.0, 1.0, 0.05):
            timings = do_bench_detailed(lambda: sparse_fn(p))
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

    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True, parents=True)

    for breakpoint_name in df["breakpoint"].unique():
        fig, ax = plt.subplots(figsize=(5, 4))

        sub_df = df[df["breakpoint"] == breakpoint_name]
        for i, (name, item) in enumerate(dense_data.items()):
            median = item[breakpoint_name]["median"]
            ax.axhline(median, label=name, linestyle="--", color=f"C{i+1}")

        sns.lineplot(
            data=sub_df,
            x="sparsity",
            y="time",
            hue="Method",
            marker="o",
            markersize=4,
            estimator=np.median,
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
            log_dir / f"matmul-detailed_m{M}n{N}k{K}_{breakpoint_name}.png", dpi=300
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
            estimator=np.median,
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
            log_dir / f"matmul-detailed_m{M}n{N}k{K}_{breakpoint_name}_flops.png",
            dpi=300,
        )
