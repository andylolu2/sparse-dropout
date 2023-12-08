import numpy as np
import torch
import torch.types
import triton
import triton.language as tl

from flash_dropout.functional.utils import (
    blockwise_dropout_mask,
    config_product,
    mask_to_increment_table,
    min_dtype,
    threadblock_swizzle,
)
from flash_dropout.types import size


@triton.autotune(
    configs=config_product(
        num_warps=[1],
        num_stages=[1],
        BLOCK_N=[32, 64, 128],
        GROUP_M=[8],
    ),
    key=["M", "N", "K"],
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
    }
)
@triton.jit
def blockwise_dsd_matmul_kernel(
    # fmt: off
    # Tensors
    a_ptr, stride_am, stride_ak,
    b_ptr, stride_bk, stride_bn,
    c_ptr, stride_cm, stride_cn,
    table_ptr, stride_t, 
    header_ptr, stride_hm, stride_hw,
    # Other parameters
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    scale: tl.constexpr,
    # Meta-parameters
    BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr, EVEN_K: tl.constexpr,
    # fmt: on
):
    # Threadblock swizzling to improve L2 cache hit rate.
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = threadblock_swizzle(pid, grid_m, grid_n, GROUP_M)

    # Fetch the header info for this block.
    header_ptr = header_ptr + pid_m * stride_hm
    table_offset = tl.load(header_ptr + 0 * stride_hw)
    width = tl.load(header_ptr + 1 * stride_hw)
    table_ptr = table_ptr + table_offset * stride_t

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

    # Iterate to compute a block of the C matrix.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, width):
        # Advance to the next block by the increment table.
        k_inc = tl.multiple_of(tl.load(table_ptr), BLOCK_K)  # Compiler hint
        a_block_ptr = tl.advance(a_block_ptr, (0, k_inc))
        b_block_ptr = tl.advance(b_block_ptr, (k_inc, 0))

        if EVEN_K:
            a = tl.load(a_block_ptr)
            b = tl.load(b_block_ptr)
        else:  # Apply boundary checks on K
            a = tl.load(a_block_ptr, boundary_check=(1,))
            b = tl.load(b_block_ptr, boundary_check=(0,))
        a = a.to(c_ptr.dtype.element_ty)
        b = b.to(c_ptr.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=acc.dtype)

        table_ptr += 1 * stride_t
    acc *= scale
    c = acc.to(c_ptr.dtype.element_ty)

    # Store output
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def blockwise_dsd_matmul(
    A: torch.Tensor,
    table: torch.Tensor,
    header: torch.Tensor,
    B: torch.Tensor,
    BLOCK_M: int,
    BLOCK_K: int,
    scale: float = 1.0,
):
    """Compute C (dense) = scale * A (sparse) x B (dense)."""

    # Check constraints
    assert A.shape[1] == B.shape[0], "Incompatible dimensions"
    assert A.device == B.device, "Incompatible devices"

    M, K = A.shape
    K, N = B.shape

    # Allocate output
    C = torch.zeros((M, N), device=A.device, dtype=min_dtype(A.dtype, B.dtype))

    def grid(META):
        return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, META["BLOCK_N"]),)

    blockwise_dsd_matmul_kernel[grid](
        # fmt: off
        A, A.stride(0), A.stride(1),
        B, B.stride(0), B.stride(1),
        C, C.stride(0), C.stride(1),
        table, table.stride(0),
        header, header.stride(0), header.stride(1),
        M, N, K,
        BLOCK_M, BLOCK_K,
        scale,
        # fmt: on
    )
    return C


@triton.autotune(
    configs=config_product(
        num_warps=[1],
        num_stages=[1],
        BLOCK_K=[32, 64],
    ),
    key=["M", "N", "K"],
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
    }
)
@triton.jit
def blockwise_sdd_matmul_kernel(
    # fmt: off
    # Tensors
    a_ptr, stride_am, stride_ak,
    b_ptr, stride_bk, stride_bn,
    c_ptr, stride_cm, stride_cn,
    table_ptr, stride_tw, stride_th,
    # Other parameters
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    scale,
    # Meta-parameters
    BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr,
    # fmt: on
):
    pid = tl.program_id(axis=0)
    pid_m = tl.load(table_ptr + pid * stride_tw + 0 * stride_th)
    pid_n = tl.load(table_ptr + pid * stride_tw + 1 * stride_th)

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

    # Iterate to compute a block of the C matrix.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        if EVEN_K:
            a = tl.load(a_block_ptr)
            b = tl.load(b_block_ptr)
        else:  # Apply boundary checks
            a = tl.load(a_block_ptr, boundary_check=(1,))
            b = tl.load(b_block_ptr, boundary_check=(0,))
        a = a.to(c_ptr.dtype.element_ty)
        b = b.to(c_ptr.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=acc.dtype)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    acc *= scale
    c = acc.to(c_ptr.dtype.element_ty)

    # Store output
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def blockwise_sdd_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    table: torch.Tensor,
    BLOCK_M: int,
    BLOCK_N: int,
    scale: float = 1.0,
):
    """Compute C (sparse) = scale * A (dense) x B (dense)."""

    # Check constraints
    assert A.shape[1] == B.shape[0], "Incompatible dimensions"
    assert A.device == B.device, "Incompatible devices"

    M, K = A.shape
    K, N = B.shape

    # Allocate output
    C = torch.zeros((M, N), device=A.device, dtype=min_dtype(A.dtype, B.dtype))
    # table = np.stack(np.nonzero(~mask), axis=1)
    # table = torch.from_numpy(table).to(A.device)
    # table = torch.nonzero(~mask, as_tuple=False).to(A.device)

    def grid(META):
        return (len(table),)

    blockwise_sdd_matmul_kernel[grid](
        # fmt: off
        A, A.stride(0), A.stride(1),
        B, B.stride(0), B.stride(1),
        C, C.stride(0), C.stride(1),
        table, table.stride(0), table.stride(1),
        M, N, K,
        BLOCK_M, BLOCK_N,
        scale,
        # fmt: on
    )
    return C


class BlockwiseDropoutMatmul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        block_size: size,
        p: float,
    ):
        assert 0 <= p < 1, "Dropout probability must be in [0, 1)"

        BLOCK_M, BLOCK_K = block_size
        mask = blockwise_dropout_mask(input, (BLOCK_M, BLOCK_K), p=p)

        ctx.save_for_backward(input, weight, mask)
        ctx.block_size = block_size
        ctx.p = p

        # y (MN dense) = x (MK sparse) x w (KN dense)
        table, offsets, widths = mask_to_increment_table(mask, BLOCK_K)
        # Pack offsets and widths into a single tensor. Helps improve cache locality.
        header = torch.stack((offsets, widths), dim=1)

        result = blockwise_dsd_matmul(
            input, table, header, weight, BLOCK_M, BLOCK_K, scale=1 / (1 - p)
        )
        return result

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ):
        input, weight, mask = ctx.saved_tensors
        BLOCK_M, BLOCK_K = ctx.block_size
        p = ctx.p

        # dx (MK sparse) = dy (MN dense) x w^T (KN dense)
        table = torch.nonzero(~mask, as_tuple=False)
        grad_input = blockwise_sdd_matmul(
            grad_output, weight.T, table, BLOCK_M, BLOCK_K, scale=1 / (1 - p)
        )

        # dw (KN dense) = x^T (KM sparse) x dy (MN dense)
        table, offsets, widths = mask_to_increment_table(mask.T, BLOCK_M)
        header = torch.stack((offsets, widths), dim=1)
        grad_weight = blockwise_dsd_matmul(
            input.T, table, header, grad_output, BLOCK_K, BLOCK_M, scale=1 / (1 - p)
        )

        return grad_input, grad_weight, None, None


def blockwise_dropout_matmul(
    input: torch.Tensor, weight: torch.Tensor, block_size: size, p: float
) -> torch.Tensor:
    return BlockwiseDropoutMatmul.apply(input, weight, block_size, p)  # type: ignore
