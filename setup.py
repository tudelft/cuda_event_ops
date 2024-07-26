from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="cuda_3d_ops",
    ext_modules=[
        CUDAExtension(
            "cuda_3d_ops",
            ["iterative_3d_warp/extension.cpp", "iterative_3d_warp/kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
