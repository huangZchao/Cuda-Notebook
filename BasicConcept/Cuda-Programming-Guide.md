# Kernels
即OP，
## 向量加样例
``` C++
∕∕ Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}
int main()
{
    ...
    ∕∕ Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    ...
}
```
## 矩阵加（单个thread block）
``` C++
∕∕ Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}
int main()
{
    ...
    ∕∕ Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

## 矩阵加（多个thread block）
``` C++
∕∕ Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
    C[i][j] = A[i][j] + B[i][j];
}
int main()
{
    ...
    ∕∕ Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N ∕ threadsPerBlock.x, N ∕ threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

