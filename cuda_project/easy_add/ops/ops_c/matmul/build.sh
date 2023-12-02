#nvcc -o matmul matmul_cuda.cu
nvcc -o matmul_v2 matmul_cuda_v2.cu -lcublas
