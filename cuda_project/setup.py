from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='CudaDemo',
    packages=find_packages(),
    version='0.1.0',
    author='xxx',

    ext_modules=[
        CUDAExtension(
            'sum_double',
            ['./ops/src/sum_two_arrays/two_sum.cpp',
             './ops/ops_c/sum_two_arrs/two_sum_cuda.cu',]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }    
)