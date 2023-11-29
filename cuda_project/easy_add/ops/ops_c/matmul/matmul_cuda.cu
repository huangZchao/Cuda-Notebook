#include <cstdio>
#include<iostream>

#define THREAD_X_PER_BLOCK 16
#define THREAD_Y_PER_BLOCK 16
#define WARP 32
#define DIVUP(m, n) ((m+n-1)/n)

__global__ void matmul_v1_kernel(const float* a, const float* b, float* c, int m, int k, int n) {
    int nrow = blockIdx.x * blockDim.x + threadIdx.x;
    int ncol = blockIdx.y * blockDim.y + threadIdx.y;

    if (nrow < m && ncol < n) {
        for(int i = 0; i < k; ++i) {
            c[nrow * n + ncol] += a[nrow * k + i] * b[i * n + ncol];
        }
    }
}

void matmul_v1_launcher(const float* a, const float* b, float* c, int m, int k, int n) {
    dim3 blockSize(DIVUP(m, THREAD_X_PER_BLOCK), DIVUP(n, THREAD_Y_PER_BLOCK));
    dim3 threadSize(THREAD_X_PER_BLOCK, THREAD_Y_PER_BLOCK);
    matmul_v1_kernel<<<blockSize, threadSize>>>(a, b, c, m, k, n);
}

void print_mat(const float* a, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
			std::cout<<a[i * n + j]<<" ";
        }
		std::cout<<std::endl;
    }
}

int main() {
    int M = 2, N = 2, K = 2;

    float *x1, *x2, *y, *d_x1, *d_x2, *d_y;
    x1 = (float*)malloc(M*K*sizeof(float));
    x2 = (float*)malloc(K*N*sizeof(float));
    y = (float*)malloc(M*N*sizeof(float));
  
    cudaMalloc(&d_x1, M*K*sizeof(float)); 
    cudaMalloc(&d_x2, K*N*sizeof(float));
    cudaMalloc(&d_y, M*N*sizeof(float));
  
    for (int i = 0; i < M * K; i++) {
        x1[i] = 1.0f;
    }
    print_mat(x1, M, K);
    for (int i = 0; i < K * N; i++) {
        x2[i] = 1.0f;
    } 
    print_mat(x2, K, N);
  
    cudaMemcpy(d_x1, x1, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, x2, K*N*sizeof(float), cudaMemcpyHostToDevice);
  
    // Perform SAXPY on 1M elements
    matmul_v1_launcher(d_x1, d_x2, d_y, M, K, N);
  
    cudaMemcpy(y, d_y, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    print_mat(y, M, N);

  
    cudaFree(d_x1);
    cudaFree(d_x2);
    cudaFree(d_y);    
    free(x1);
    free(x2);
    free(y);

    return 0;
}