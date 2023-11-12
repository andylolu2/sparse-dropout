import numpy as np
import torch
import torch.types
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"GROUP_SIZE_M": 8},
            num_stages=1,
            num_warps=4,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel_with_block_pointers(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_table_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    K_table,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_a_table_m,
    stride_a_table_k,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See the matrix multiplication tutorial for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create block pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction and accumulate.
    # See above `Make a Block Pointer` section for details.
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0),
    )
    a_table_ptr = a_table_ptr + pid_m * stride_a_table_m

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block.
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, K_table):
        k_inc = tl.load(a_table_ptr)
        k_inc = tl.multiple_of(k_inc, 8)

        # Advance the block pointer to the next K block.
        # See above `Advance a Block Pointer` section for details.
        a_block_ptr = tl.advance(a_block_ptr, (0, k_inc))
        b_block_ptr = tl.advance(b_block_ptr, (k_inc, 0))

        # Load with boundary checks, no need to calculate the mask manually.
        # For better performance, you may remove some axis from the boundary
        # check, if you can guarantee that the access is always in-bound in
        # that axis.
        # See above `Load/Store a Block Pointer` section for details.
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))

        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        a_table_ptr += stride_a_table_k

    c = accumulator.to(tl.float16)

    # ----------------------------------------------------------------
    # Write back the block of the output matrix C with boundary checks.
    # See above `Load/Store a Block Pointer` section for details.
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def dropout_mask(x: torch.Tensor, block_size: tuple[int, int], p: float):
    """mask = True means dropped."""
    assert x.ndim == 2

    num_blocks = (x.shape[0] // block_size[0], x.shape[1] // block_size[1])
    num_blocks_not_masked = round(num_blocks[1] * (1 - p))

    not_masked_indices = np.argsort(np.random.rand(*num_blocks), axis=1)
    not_masked_indices = np.sort(not_masked_indices[:, :num_blocks_not_masked], axis=1)
    return not_masked_indices


def mask_to_increment_table(mask: np.ndarray, block_size: int):
    """Example:

    block_size = 16
    mask = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ]

    table = [
        [0,  16, 16],
        [0,  16, 32],
        [0,  32, 16],
        [16, 16, 16],
    ]
    """
    mask = np.concatenate(
        (np.zeros((mask.shape[0], 1), dtype=mask.dtype), mask), axis=-1
    )
    table = np.diff(mask) * block_size
    return table


def matmul(a: torch.Tensor, b: torch.Tensor):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape

    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_SIZE = (64, 64, 64)  # M N K

    # 2D launch kernel where each block gets its own program.
    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    mask = dropout_mask(a, (BLOCK_SIZE[0], BLOCK_SIZE[2]), p=0.2)
    table = mask_to_increment_table(mask, BLOCK_SIZE[1])
    table = torch.from_numpy(table).to(a.device)

    matmul_kernel_with_block_pointers[grid](
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
        BLOCK_SIZE[0],
        BLOCK_SIZE[1],
        BLOCK_SIZE[2],
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
