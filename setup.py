"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

import paddle
from site import getsitepackages
paddle_includes = []
for site_packages_path in getsitepackages():
    paddle_includes.append(os.path.join(site_packages_path, "paddle", "include"))
    paddle_includes.append(os.path.join(site_packages_path, "paddle", "include", "third_party"))
    paddle_includes.append(os.path.join(site_packages_path, "nvidia", "cudnn", "include"))

from paddle.utils.cpp_extension import CUDAExtension, setup

# Compiler flags.
CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--use_fast_math",
    "--threads=8",
    "-Xptxas=-v",
    "-diag-suppress=174", # suppress the specific warning
]

ABI = 1
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

prop = paddle.device.cuda.get_device_properties()
cc = prop.major * 10 + prop.minor
NVCC_FLAGS += ["-gencode", "arch=compute_{0},code=sm_{0}".format(cc)]

ext_modules = []

# Attention kernels.
qattn_fused_extension = CUDAExtension(
    sources=[
        "csrc/qattn_fused/pybind.cpp",
        "csrc/qattn_fused/fused.cu",
        "csrc/qattn_fused/qk_int_sv_f16_per_warp_cuda.cu",
        "csrc/qattn_fused/qk_int_sv_f8_per_warp_cuda.cu",
        "csrc/qattn_fused/qk_int_sv_f16_per_warp_buffer_cuda.cu",
        "csrc/qattn_fused/qk_int_sv_f8_per_warp_buffer_cuda.cu",
    ],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
    include_dirs=paddle_includes,
)

setup(
    name='qattn_fused', 
    ext_modules=qattn_fused_extension,
)