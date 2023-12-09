from torch.utils.cpp_extension import load

# load the compiled extension module
fdropout = load(
    name="fdropout",
    sources=[
        "flash_dropout/cuda/src/fdropout.cu",
    ],
    extra_include_paths=[
        "flash_dropout/cuda/cutlass/include",
    ],
)

if __name__ == "__main__":
    import torch

    x = torch.arange(8, dtype=torch.float32, device="cuda").reshape(2, 4)
    print(x)

    (y,) = fdropout.forward(x)
    print(y)

    (dx,) = fdropout.backward(y)
    print(dx)
