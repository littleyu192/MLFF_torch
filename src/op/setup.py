from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# import sys
# sys.path.append("/home/husiyu/tools/cuda-11/bin")

setup(
    name="op",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            name="op",
            sources=[
                "calculate_force.cpp",
                "kernel/calculateForce.cu",
                "calculate_force_grad.cpp",
                "kernel/calculateForceGrad.cu",
                "calculate_virial_force.cpp",
                "kernel/calculateVirial.cu",
                "calculate_virial_force_grad.cpp",
                "kernel/calculateVirialGrad.cu",
                "calculate_DR.cpp",
                "kernel/calculateDR.cu",
                "register_op.cpp",
                "kernel/cutlass_gemm_array.cu",
                "matmul_bias_tanh.cpp",
            ],
            extra_compile_args={"nvcc": ["-std=c++17"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
