import lightning as L
import torch
import torch.types
import triton
import triton.testing

from flash_dropout.functional.blockwise_dropout_matmul_cuda import (
    blockwise_dropout_matmul as blockwise_dropout_matmul_cuda,
)
from flash_dropout.functional.blockwise_dropout_matmul_triton import (
    blockwise_dropout_matmul,
    blockwise_dsd_matmul,
    blockwise_sdd_matmul,
)
from flash_dropout.functional.naive import (
    blockwise_dropout_matmul as naive_blockwise_dropout_matmul,
)
from flash_dropout.functional.utils import (
    blockwise_dropout_mask,
    mask_to_increment_table,
)

block_size = (64, 64)
p = 0.5
cache = {}


def f_triton(A: torch.Tensor, B: torch.Tensor):
    C = blockwise_dropout_matmul(A, B, block_size, p)
    C.backward(torch.zeros_like(C))


def f_triton_cached_mask(A: torch.Tensor, B: torch.Tensor):
    global cache

    BLOCK_M, BLOCK_K = block_size
    if (A.shape, B.shape) not in cache:
        mask = blockwise_dropout_mask(A, block_size, p=p)
        fwd_table, offsets, widths = mask_to_increment_table(mask, BLOCK_K)
        fwd_header = torch.stack((offsets, widths), dim=1)

        bwd_table_1 = torch.nonzero(~mask, as_tuple=False)

        bwd_table_2, offsets, widths = mask_to_increment_table(mask.T, BLOCK_M)
        bwd_header_2 = torch.stack((offsets, widths), dim=1)
        cache[(A.shape, B.shape)] = (
            fwd_table,
            fwd_header,
            bwd_table_1,
            bwd_table_2,
            bwd_header_2,
        )
    fwd_table, fwd_header, bwd_table_1, bwd_table_2, bwd_header_2 = cache[
        (A.shape, B.shape)
    ]

    C = blockwise_dsd_matmul(
        A, fwd_table, fwd_header, B, BLOCK_M, BLOCK_K, scale=1 / (1 - p)
    )
    # dC = torch.zeros_like(C)
    # dA = blockwise_sdd_matmul(dC, B.T, bwd_table_1, BLOCK_M, BLOCK_K, scale=1 / (1 - p))
    # dB = blockwise_dsd_matmul(
    #     A.T, bwd_table_2, bwd_header_2, dC, BLOCK_K, BLOCK_M, scale=1 / (1 - p)
    # )


def f_naive(A: torch.Tensor, B: torch.Tensor):
    C = naive_blockwise_dropout_matmul(A, B, block_size, p)
    C.backward(torch.zeros_like(C))


def f_dense(A: torch.Tensor, B: torch.Tensor):
    C = A @ B
    C.backward(torch.zeros_like(C))


def f_cuda(A: torch.Tensor, B: torch.Tensor):
    C = blockwise_dropout_matmul_cuda(A, B, block_size, p)
    C.backward(torch.zeros_like(C))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=range(256, 1024 + 1, 256),
        line_arg="provider",
        line_vals=[
            "torch",
            # "triton",
            # "triton_cached",
            "dense",
            "cuda",
        ],
        line_names=[
            "PyTorch",
            # "Triton",
            # "Triton (cached)",
            "Dense",
            "CUDA",
        ],
        styles=[
            ("green", "-"),
            # ("blue", "-"),
            # ("cyan", "-"),
            ("red", "-"),
            ("black", "-"),
        ],
        ylabel="TFLOPS",
        plot_name="matmul-performance",
        args={},
    )
)
def benchmark(M, N, K, provider):
    A = torch.randn((M, K), device="cuda", dtype=torch.float16, requires_grad=True)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16, requires_grad=True)

    fs = {
        "torch": f_naive,
        "triton": f_triton,
        "triton_cached": f_triton_cached_mask,
        "dense": f_dense,
        "cuda": f_cuda,
    }

    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: fs[provider](A, B), quantiles=[0.5, 0.05, 0.95]
    )

    def perf(ms):
        return 3 * 2 * M * N * K * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    L.seed_everything(0)
    benchmark.run(save_path="./logs", print_data=True)
