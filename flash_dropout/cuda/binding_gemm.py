import torch
from torch.utils.cpp_extension import load


class GEMM:
    def __init__(self):
        """Load and JIT compile the extension module"""
        self.ext = load(
            name="gemm",
            sources=[
                "flash_dropout/cuda/src/gemm_binding.cu",
            ],
            extra_include_paths=[
                "flash_dropout/cuda/cutlass/include",
                "flash_dropout/cuda/cutlass/tools/util/include",
            ],
            extra_cuda_cflags=[
                "-std=c++20",
                "-O3",
                "--threads",
                "0",
                "--Werror",
                "all-warnings",
                "--verbose",
            ],
            extra_cflags=[
                "-std=c++20",
                "-O3",
                "-Wall",
                "-Wextra",
                "--verbose",
            ],
            verbose=True,
        )

    def gemm(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.ext.gemm(A, B)  # type: ignore


if __name__ == "__main__":
    import lightning as L

    L.seed_everything(0)
    torch.set_printoptions(sci_mode=False, edgeitems=5, linewidth=500, precision=2)

    def make_tensor(m: int, n: int, row_major=True):
        if row_major:
            return torch.randn(m, n, dtype=torch.float16, device="cuda")
        else:  # column major
            return torch.randn(n, m, dtype=torch.float16, device="cuda").T

    ext = GEMM()

    def test(A: torch.Tensor, B: torch.Tensor):
        C = ext.gemm(A, B).float()
        C_ref = A.float() @ B.float().T
        # print(f"{C=}")
        # print(f"{C_ref=}")

        abs_err_pct = (torch.abs(C - C_ref) > 1e-2).float().mean().item()
        rel_err_pct = (
            ((torch.abs(C - C_ref) / torch.abs(C_ref)) > 0.01).float().mean().item()
        )
        max_abs_err = torch.abs(C - C_ref).max().item()
        max_rel_err = (torch.abs(C - C_ref) / torch.abs(C_ref)).max().item()

        print(f"Abs err: {abs_err_pct:.2%} Rel err: {rel_err_pct:.2%}")
        print(f"Max abs err: {max_abs_err:.2e} Max rel err: {max_rel_err:.2e}")

    M, N, K = 512, 512, 512

    for a_row_major, b_row_major in [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]:
        A = make_tensor(M, K, row_major=a_row_major)
        B = make_tensor(N, K, row_major=b_row_major)
        print(f"A ({A.stride()}) B ({B.stride()})")
        test(A, B)
        print()
