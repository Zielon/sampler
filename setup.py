import os
import os.path as osp
import re
import subprocess

from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NAME = 'tetra_sampler'
DESCRIPTION = 'Tetrahedra mesh sampler'
URL = ''
EMAIL = 'wzielonka'
AUTHOR = 'Wojciech Zielonka'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '1.0.5'

here = os.path.abspath(os.path.dirname(__file__))

about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

nvcc_args = ["-O3"]
nvcc_args.extend(
    [
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_90,code=sm_90",
    ]
)

if __name__ == "__main__":
    cxx_args = []
    nvcc_args = ["--use_fast_math"] + nvcc_args
    include_dirs = [f"{here}/include"] + torch.utils.cpp_extension.include_paths()

    setup(
        name=NAME,
        description=DESCRIPTION,
        version=VERSION,
        author=AUTHOR,
        author_email=EMAIL,
        python_requires=REQUIRES_PYTHON,
        url=URL,
        packages=find_packages(),
        ext_modules=[
            CUDAExtension(
                "sampling_segments_cuda",
                sources=[
                    "src/sampler_segments.cpp",
                    "src/sampler_segmetns_kernel.cu",
                ],
                include_dirs=include_dirs,
                extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
            ),
            CUDAExtension(
                "sampling_points_cuda",
                sources=[
                    "src/sampler_points.cpp",
                    "src/sampler_points_kernel.cu",
                ],
                include_dirs=include_dirs,
                extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
            ),
        ],
        cmdclass={'build_ext': BuildExtension},
    )
