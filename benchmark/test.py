import numpy as np
import torch
import torch.types
import triton
import triton.testing

from flash_dropout.functional.blockwise_dropout_matmul_triton import (
    blockwise_dropout_matmul,
)
from flash_dropout.functional.naive import (
    blockwise_dropout_matmul as naive_blockwise_dropout_matmul,
)


def f_triton(A: torch.Tensor, B: torch.Tensor):
    C = blockwise_dropout_matmul(A, B, (64, 64), 0.5)
    C.backward(torch.zeros_like(C))


def f_naive(A: torch.Tensor, B: torch.Tensor):
    C = naive_blockwise_dropout_matmul(A, B, (64, 64), 0.5)
    C.backward(torch.zeros_like(C))


def f_dense(A: torch.Tensor, B: torch.Tensor):
    C = A @ B
    C.backward(torch.zeros_like(C))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[i for i in range(256, 4097, 256)],
        line_arg="provider",
        line_vals=["torch", "triton", "dense"],
        line_names=["PyTorch", "Triton", "Dense"],
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-performance",
        args={},
    )
)
def benchmark(M, N, K, provider):
    A = torch.randn((M, K), device="cuda", dtype=torch.float16, requires_grad=True)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16, requires_grad=True)

    fs = {"torch": f_naive, "triton": f_triton, "dense": f_dense}

    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: fs[provider](A, B), quantiles=[0.5, 0.05, 0.95]
    )

    def perf(ms):
        return 3 * 2 * M * N * K * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    benchmark.run(save_path="./logs", print_data=True)
