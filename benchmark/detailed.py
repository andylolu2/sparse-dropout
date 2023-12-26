import math

import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.types

from flash_dropout.cuda.binding import FlashDropoutCUDA
from flash_dropout.functional.blockwise_dropout_matmul_triton import (
    blockwise_dropout_matmul,
    blockwise_dsd_matmul,
    blockwise_sdd_matmul,
)
from flash_dropout.functional.naive import blockwise_dropout
from flash_dropout.functional.naive import (
    blockwise_dropout_matmul as naive_blockwise_dropout_matmul,
)
from flash_dropout.functional.utils import (
    blockwise_dropout_mask,
    mask_to_increment_table,
)

block_size = (256, 128)


def f_naive(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor, p: float):
    mask = blockwise_dropout_mask(A, block_size, p)
    A = blockwise_dropout(A, mask, block_size, p)
    C = A @ B.T
    yield C

    dA = blockwise_dropout(dC @ B.T, mask, block_size, p)
    yield dA

    dB = dC.T @ A
    yield dB


def f_dense(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor, p: float):
    yield A @ B.T
    yield dC @ B
    yield dC.T @ A


def f_baseline(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor, p: float):
    A_ = A.clone().requires_grad_(True)
    B_ = B.clone().requires_grad_(True)

    A__ = A_
    A__ = torch.dropout(A_, p, train=True)
    C = A__ @ B_.T
    yield C

    C.backward(dC)
    yield A_.grad
    yield B_.grad


def f_cuda(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor, p: float):
    impl = FlashDropoutCUDA(
        BLK_MNK_GROUP_0=(128, 128, 32, 5),
        BLK_MNK_GROUP_1=(128, 32, 128, 5),
        BLK_MNK_GROUP_2=(32, 128, 128, 5),
    )

    C, mask, mask_T, mask_table, count = impl.forward(A, B, p)
    yield C
    yield impl.backward_dA(dC, B, mask_table, p, count)
    yield impl.backward_dB(dC, A, mask_T, p)


def do_bench_detailed(fn, warmup=100, rep=500):
    n_breakpoints = 0
    for _ in fn():
        n_breakpoints += 1
    torch.cuda.synchronize()

    cache = torch.empty(int(4e6 // 8), dtype=torch.int64, device="cuda")

    # Estimate the runtime of the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(10):
        cache.zero_()
        for _ in fn():
            pass
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 10

    # compute number of warmup and repeat
    n_warmup = math.ceil(warmup / estimate_ms)
    n_repeat = math.ceil(rep / estimate_ms)

    events = [
        [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(n_breakpoints)
        ]
        for _ in range(n_repeat)
    ]

    for _ in range(n_warmup):
        for _ in fn():
            pass

    # Benchmark
    for breakpoint_events in events:
        cache.zero_()

        f = iter(fn())
        for start, end in breakpoint_events:
            start.record()
            next(f)
            end.record()

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

    M, N, K = 128 * 16, 512 * 4, 512
    M, N, K = 128 * 16, 512, 512 * 4
    # M, N, K = 4096, 4096, 4096

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    dC = torch.randn((M, N), device="cuda", dtype=torch.float16)

    fs = {
        "dense": lambda: f_dense(A, B, dC, 0.5),
        "baseline": lambda: f_baseline(A, B, dC, 0.5),
        "p=0.0": lambda: f_cuda(A, B, dC, 0.0),
        "p=0.25": lambda: f_cuda(A, B, dC, 0.25),
        "p=0.5": lambda: f_cuda(A, B, dC, 0.5),
        "p=0.75": lambda: f_cuda(A, B, dC, 0.75),
        "p=0.95": lambda: f_cuda(A, B, dC, 0.95),
    }

    data = []

    for name, fn in fs.items():
        timings = do_bench_detailed(fn)
        for i, timing in enumerate(timings):
            for t in timing:
                data.append({"name": name, "breakpoint": i, "time": t.item()})

    df = pd.DataFrame(data)

    sns.barplot(df, x="breakpoint", y="time", hue="name")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("./logs/matmul-detailed_8.png", dpi=300)
