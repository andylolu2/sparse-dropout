import os
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = Path(__file__).resolve().parent
cuda_dir = this_dir / "flash_dropout" / "cuda"

generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_75,code=sm_75")

nvcc_flag = ["--threads", "4"]

torch._C._GLIBCXX_USE_CXX11_ABI = False

setup(
    name="fdropout",
    ext_modules=[
        CUDAExtension(
            name="fdropout",
            sources=[
                "flash_dropout/cuda/src/fdropout.cu",
            ],
            include_dirs=[
                cuda_dir / "cutlass" / "include",
                cuda_dir / "cutlass" / "tools" / "util" / "include",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + generator_flag,
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    *nvcc_flag,
                    *generator_flag,
                    *cc_flag,
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
