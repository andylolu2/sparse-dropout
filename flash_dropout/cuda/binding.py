import torch
from torch.utils.cpp_extension import load

# load the compiled extension module
fdropout = load(
    name="fdropout",
    sources=[
        "flash_dropout/cuda/src/fdropout.cu",
    ],
    extra_include_paths=[
        "flash_dropout/cuda/cutlass/include",
        "flash_dropout/cuda/cutlass/tools/util/include",
    ],
)

if __name__ == "__main__":
    M, N, K = 128, 128, 128

    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(K, N, dtype=torch.float32, device="cuda")

    (C,) = fdropout.forward(A, B)

    print(C)
