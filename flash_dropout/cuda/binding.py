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
