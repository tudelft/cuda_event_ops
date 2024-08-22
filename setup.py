from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


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
