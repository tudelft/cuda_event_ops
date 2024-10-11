from setuptools import find_packages, setup
import os

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# set cuda arch if cross-compiling and gpu not available (e.g. during docker build)
# https://en.wikipedia.org/wiki/CUDA#GPUs_supported
if not torch.cuda.is_available():
    if os.environ.get("TORCH_CUDA_ARCH_LIST") is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;8.7;8.9+PTX"  # turing, ampere, ada


setup(
    name="cuda_3d_ops",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            "iterative_3d_warp_cuda._C",
            ["iterative_3d_warp/extension.cpp", "iterative_3d_warp/kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
        CUDAExtension(
            "trilinear_splat_cuda._C",
            ["trilinear_splat/extension.cpp", "trilinear_splat/kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
