import torch
from torch.utils.cpp_extension import load

# load and JIT compile the extension module
fdropout = load(
    name="fdropout",
    sources=[
        "flash_dropout/cuda/src/fdropout.cu",
    ],
    extra_include_paths=[
        "flash_dropout/cuda/cutlass/include",
        "flash_dropout/cuda/cutlass/tools/util/include",
    ],
    extra_cflags=["-std=c++17", "-O3"],
    extra_cuda_cflags=["-std=c++17", "-O3", "--threads", "8"],
    verbose=True,
)

# --- Wrappers for the extension ---


def forward(
    A: torch.Tensor, B: torch.Tensor, p: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    C, mask, mask_T, mask_table, count = fdropout.forward(A, B, p)
    return C, mask, mask_T, mask_table, count


def forward_test(
    A: torch.Tensor, B: torch.Tensor, m: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    C, mask, mask_T, mask_table, count = fdropout.forward_test(A, B, m)
    return C, mask, mask_T, mask_table, count


if __name__ == "__main__":
    import lightning as L

    L.seed_everything(0)
    torch.set_printoptions(sci_mode=False, edgeitems=5, linewidth=120)

    M, N, K = 1024, 1024, 1024

    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    p = 0.0

    C, mask, mask_T, mask_table, count = forward(A, B, p)

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
