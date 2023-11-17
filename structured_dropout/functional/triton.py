import numpy as np
import torch
import torch.types
import triton
import triton.language as tl

from .utils import dropout_mask, mask_to_increment_table


@triton.jit
def threadblock_swizzle(pid, grid_m, grid_n, GROUP_M: tl.constexpr):
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    return pid_m, pid_n


@triton.autotune(
    configs=[
        triton.Config({"GROUP_M": 8}, num_stages=1, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
    }
)
@triton.jit
def blockwise_dropout_matmul_kernel(
    # Pointers to matrices
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    c_ptr: tl.tensor,
    a_table_ptr: tl.tensor,
    # Matrix dimensions
    M,
    N,
    K,
    K_table,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_a_table_m,
    stride_a_table_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # Meta-parameters
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Kernel for computing the blockwise sparse matmul C = A x B.

    A (sparse) has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    # Threadblock swizzling to improve L2 cache hit rate.
    # Note: using a 1D launch grid seems to be better than a 2D one.
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = threadblock_swizzle(pid, grid_m, grid_n, GROUP_M)

    # Create block pointers for blocks of A and B.
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    # Pointer to the m-th row of the increment table.
    a_table_ptr = a_table_ptr + pid_m * stride_a_table_m

    # Iterate to compute a block of the C matrix.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K_table):
        # Advance the block pointer to the next K block by the increment table.
        k_inc = tl.load(a_table_ptr)
        k_inc = tl.multiple_of(k_inc, BLOCK_K)  # Compiler hint
        a_block_ptr = tl.advance(a_block_ptr, (0, k_inc))
        b_block_ptr = tl.advance(b_block_ptr, (k_inc, 0))

        if EVEN_K:
            a = tl.load(a_block_ptr)
            b = tl.load(b_block_ptr)
        else:  # Apply boundary checks.
            a = tl.load(a_block_ptr, boundary_check=(0, 1))
            b = tl.load(b_block_ptr, boundary_check=(0, 1))

        # We accumulate along the K dimension.
        acc += tl.dot(a, b)
        a_table_ptr += stride_a_table_k
    c = acc.to(tl.float16)

    # Store output
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    if EVEN_K:
        tl.store(c_block_ptr, c)
    else:
        tl.store(c_block_ptr, c, boundary_check=(0, 1))


def matmul(a: torch.Tensor, b: torch.Tensor):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape

    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64

    # 2D launch kernel where each block gets its own program.
    def grid(META):
        return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    mask = dropout_mask(a, (BLOCK_M, BLOCK_K), p=0.2)
    table = mask_to_increment_table(mask, BLOCK_K)
    table = torch.from_numpy(table).to(a.device)

    blockwise_dropout_matmul_kernel[grid](
        a,
        b,
        c,
        table,
        M,
        N,
        K,
        table.shape[1],
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        table.stride(0),
        table.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )
    return c


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(2, 33)
        ],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=["cublas", "triton"],
        # Label name for the lines
        line_names=["cuBLAS", "Triton"],
        # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "cublas":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b), quantiles=quantiles
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b), quantiles=quantiles
        )
    else:
        raise ValueError()

    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    a = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((512, 512), device="cuda", dtype=torch.float16)

    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)

    print(f"{triton_output=}")
    print(f"{torch_output=}")

    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

    benchmark.run(show_plots=True, print_data=True)
