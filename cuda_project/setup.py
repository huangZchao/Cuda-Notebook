from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='CudaDemo',
    packages=find_packages(),
    version='0.1.0',
    author='hzc',

    ext_modules=[
        CUDAExtension(
            'sum_double',
            ['./ops/ops_c/sum_two_arrs/two_sum.cpp',
             './ops/ops_c/sum_two_arrs/two_sum_cuda.cu',]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }    
)