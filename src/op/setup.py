from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

cuda_source = glob.glob("**/*.cu", recursive=True)
cpp_source = glob.glob("**/*.cpp", recursive=True)

setup(
    name="op",
    version="0.1.0",
    author="Siyu Hu",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "op",
            cuda_source + cpp_source,
            extra_compile_args={"nvcc": ["-std=c++17"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch",
    ],
)
