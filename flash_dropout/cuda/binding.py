from functools import lru_cache

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=None)
class FlashDropoutCUDA:
    def __init__(
        self,
        BLK_MNK_GROUP_0: tuple[int, int, int, int],
        BLK_MNK_GROUP_1: tuple[int, int, int, int],
        BLK_MNK_GROUP_2: tuple[int, int, int, int],
    ):
        """Load and JIT compile the extension module"""
        BLK_M_0, BLK_N_0, BLK_K_0, GROUP_0 = BLK_MNK_GROUP_0
        BLK_M_1, BLK_N_1, BLK_K_1, GROUP_1 = BLK_MNK_GROUP_1
        BLK_M_2, BLK_N_2, BLK_K_2, GROUP_2 = BLK_MNK_GROUP_2
        self.ext = load(
            name="fdropout",
            sources=[
                "flash_dropout/cuda/src/fdropout.cu",
            ],
            extra_include_paths=[
                "flash_dropout/cuda/cutlass/include",
                "flash_dropout/cuda/cutlass/tools/util/include",
            ],
            extra_cuda_cflags=[
                "-std=c++17",
                "-O3",
                "--threads",
                "8",
                f"-DJIT_{BLK_M_0=}",
                f"-DJIT_{BLK_N_0=}",
                f"-DJIT_{BLK_K_0=}",
                f"-DJIT_{GROUP_0=}",
                f"-DJIT_{BLK_M_1=}",
                f"-DJIT_{BLK_N_1=}",
                f"-DJIT_{BLK_K_1=}",
                f"-DJIT_{GROUP_1=}",
                f"-DJIT_{BLK_M_2=}",
                f"-DJIT_{BLK_N_2=}",
                f"-DJIT_{BLK_K_2=}",
                f"-DJIT_{GROUP_2=}",
            ],
            verbose=True,
        )

    def forward(
        self, A: torch.Tensor, B: torch.Tensor, p: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            C
            mask
            mask_T
            mask_table
            count
        """
        return self.ext.forward(A, B, p)  # type: ignore

    def backward_dA(
        self,
        dC: torch.Tensor,
        B: torch.Tensor,
        mask_table: torch.Tensor,
        p: float,
        count: int,
    ) -> torch.Tensor:
        """
        Returns:
            dA
        """
        return self.ext.backward_dA(dC, B, mask_table, p, count)  # type: ignore

    def backward_dB(
        self, dC: torch.Tensor, A: torch.Tensor, mask_T: torch.Tensor, p: float
    ) -> torch.Tensor:
        """
        Returns:
            dB
        """
        return self.ext.backward_dB(dC, A, mask_T, p)  # type: ignore

    def forward_test(
        self, A: torch.Tensor, B: torch.Tensor, m: torch.Tensor, p: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            C
            mask
            mask_T
            mask_table
            count
        """
        return self.ext.forward_test(A, B, m, p)  # type: ignore


if __name__ == "__main__":
    import lightning as L

    L.seed_everything(0)
    torch.set_printoptions(sci_mode=False, edgeitems=5, linewidth=120)

    M, N, K = 512, 512, 512

    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    p = 0.0

    ext = FlashDropoutCUDA(
        BLK_MNK_GROUP_0=(128, 128, 64, 5),
        BLK_MNK_GROUP_1=(128, 64, 128, 5),
        BLK_MNK_GROUP_2=(64, 128, 128, 5),
    )

    C, mask, mask_T, mask_table, count = ext.forward(A, B, p)

    C_ref = A.float() @ B.T.float()
    C = C.float()

    max_abs_err = torch.max(torch.abs(C - C_ref))
    max_rel_err = torch.max(torch.abs(C - C_ref) / torch.abs(C_ref))

    print(f"{C=}")
    print(f"{C_ref=}")

    print(f"mask: {mask.shape}")
    for i in mask:
        i = i.item()
        v = i if i >= 0 else i + (1 << 64)
        print(f"{v:0>64b}")
    print(f"mask_T: {mask_T.shape}")
    for i in mask_T:
        i = i.item()
        v = i if i >= 0 else i + (1 << 64)
        print(f"{v:0>64b}")
    print(f"{mask_table=}")
    print(f"{count=}")

    print(f"{max_abs_err=}")
    print(f"{max_rel_err=}")
