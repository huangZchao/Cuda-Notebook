#include <cstdio>

#define THREAD_PER_BLOCK 256
#define WARP 32
#define DIVUP(m, n) ((m+n-1)/n)

__global__ void matmul_v1_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    int nrow = blockIdx.x * blockDim.x + threadIdx.x;
    int ncol = blockIdx.y * blockDim.y + threadIdx.y;

    for(int i = 0; i < k; ++i) {
        c[nrow * n + ncol] += a[nrow * k + i] * b[i * n + ncol]
    }
}

void matmul_v1_launcher(const float* a, const float* b, float* c, int m, int n, int k) {
    dim3 blockSize(DIVUP(m, THREAD_PER_BLOCK), DIVUP(n, THREAD_PER_BLOCK));
    dim3 threadSize(THREAD_PER_BLOCK);
    matmul_v1_kernel<<<blockSize, threadSize>>>(a, b, c, m, n, k);
}