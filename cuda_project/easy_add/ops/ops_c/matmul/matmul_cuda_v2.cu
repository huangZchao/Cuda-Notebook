// optimize sgemm

#include <stdio.h>
#include <stdlib.h>
#include "assert.h" 

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

template<
    const int BLOCK_SIZE_M, // height of block of C that each thread block calculate
    const int BLOCK_SIZE_K, // width of block of A and height of block of B that each thread block load into shared memory
    const int BLOCK_SIZE_N, // width of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculat
    const int THREAD_SIZE_X, // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER
>
__global__ void Sgemm(
    float* __restrict__ A,
    float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int K,
    const int N
) {
    // block indx
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // thread idx
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // thread num per block
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    // cur thread id
    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    // mem alloc
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    // register alloc
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    float reg_a[2][THREAD_SIZE_Y];
    float reg_b[2][THREAD_SIZE_X];

    // moving times for moving data from global to register
    const int ldg_num_a = (BLOCK_SIZE_M * BLOCK_SIZE_K) / (THREAD_NUM_PER_BLOCK * 4);
    const int ldg_num_b = (BLOCK_SIZE_K * BLOCK_SIZE_N) / (THREAD_NUM_PER_BLOCK * 4);
    // register each thread needed 
    float ldg_a_reg[4 * ldg_num_a]; // todo 没明白？？？？？？？？？？？
    float ldg_b_reg[4 * ldg_num_b];

    // moving thread num
    const int A_TILE_THREAD_NUM_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_NUM_ROW = BLOCK_SIZE_N / 4;

    const int A_TILE_ROW_STRIDE  = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_NUM_ROW;
    const int B_TILE_ROW_STRIDE  = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_NUM_ROW;

    const int A_TILE_ROW_START = tid / A_TILE_THREAD_NUM_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_NUM_ROW;

    const int A_TILE_COL_IDX = tid % A_TILE_THREAD_NUM_ROW * 4;
    const int B_TILE_COL_IDX = tid % B_TILE_THREAD_NUM_ROW * 4;

    A = &A[(BLOCK_SIZE_M * by)* K]; 
    B = &B[BLOCK_SIZE_N * bx];

    // transfer **first** tile from global mem to shared mem
    // load A from global memory to shared memory
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
            A_TILE_ROW_START + i,
            A_TILE_COL_IDX,
            K
        )]);
        As[0][A_TILE_COL_IDX][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index];
        As[0][A_TILE_COL_IDX + 1][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 1];
        As[0][A_TILE_COL_IDX + 2][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 2];
        As[0][A_TILE_COL_IDX + 3][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 3];        
    }
    // load B from global memory to shared memory   
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL_IDX]) = FETCH_FLOAT4(B[OFFSET(
            B_TILE_ROW_START + i,
            B_TILE_COL_IDX,
            N
        )]);    
    }
    __syncthreads();
    
    // load A from shared memory to register
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        FETCH_FLOAT4(reg_a[0][thread_y]) = FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
    }
    // load B from shared memory to register
    #pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        FETCH_FLOAT4(reg_b[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
    }    

    int write_stage_idx = 1;
    int tile_idx = 0;

    do{
        // 大迭代
        tile_idx += BLOCK_SIZE_K;
        if(tile_idx < K) {
            // load A next tile from global mem to ldg
            for(int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                    A_TILE_ROW_START + i,
                    A_TILE_COL_IDX + tile_idx,
                    K
                )]);
            }
            // load B next tile from global mem to ldg
            for(int i = 0; i < BLOCK_SIZE_M; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
                    B_TILE_ROW_START + i + tile_idx,
                    B_TILE_COL_IDX,
                    N
                )]);
            }            
        }

        int load_stage_idx = write_stage_idx ^ 1; // shared mem 0 or 1

        // 计算小迭代
        #pragma unroll
        for(int j = 0; j < BLOCK_SIZE_K - 1; ++j) {
            // load A next tile from shared mem to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
                FETCH_FLOAT4(reg_a[(j+1)%2][thread_y]) = FETCH_FLOAT4(As[load_stage_idx][j+1][THREAD_SIZE_Y * ty + thread_y]);
            }
            // load B next tile from shared mem to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                FETCH_FLOAT4(reg_b[(j+1)%2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx][j+1][THREAD_SIZE_X * tx + thread_x]);
            }  
            // compute C
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += reg_a[j%2][thread_y] * reg_b[j%2][thread_x];
                }
            }  
        }

        if(tile_idx < K){
            // load A from ldg to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                As[write_stage_idx][A_TILE_COL_IDX][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
                As[write_stage_idx][A_TILE_COL_IDX+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
                As[write_stage_idx][A_TILE_COL_IDX+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
                As[write_stage_idx][A_TILE_COL_IDX+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
            }
            // load B from ldg to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL_IDX]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            // use double buffer, only need one sync
            __syncthreads();
            // switch
            write_stage_idx ^= 1;
        }

        // load first tile from shared mem to register of next iter
        // load A from shared memory to register
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
            FETCH_FLOAT4(reg_a[0][thread_y]) = FETCH_FLOAT4(As[load_stage_idx^1][0][THREAD_SIZE_Y * ty + thread_y]);
        }
        // load B from shared memory to register
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            FETCH_FLOAT4(reg_b[0][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][THREAD_SIZE_X * tx + thread_x]);
        }
        //compute last tile mma THREAD_SIZE_X x THREAD_SIZE_Y
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += reg_a[(BLOCK_SIZE_K - 1) % 2][thread_y] * reg_b[(BLOCK_SIZE_K - 1) % 2][thread_x];
            }
        } 
    }while(tile_idx < K);   

    // store C back 
    for(int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        for(int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + THREAD_SIZE_Y * ty + thread_y,
                BLOCK_SIZE_N * bx + THREAD_SIZE_X * tx + thread_x,
                K
            )]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
        }
    }     
}

void matmul_v2_launcher(float* a, float* b, float* c, const int m, const int k, const int n) {
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    const bool ENABLE_DOUBLE_BUFFER = true;

    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid(n / BLOCK_SIZE_N, m / BLOCK_SIZE_M);
    Sgemm<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> 
    <<< dimGrid, dimBlock >>>(a, b, c, m, k, n);
}

// int main(int argc, char** argv) {
//     if (argc != 4) {
//         printf("usage: ./main [M] [K] [N]\n");
//         exit(0);
//     }
//     size_t M = atoi(argv[1]);
//     size_t K = atoi(argv[2]);
//     size_t N = atoi(argv[3]);

//     assert( M%8 == 0); 
//     assert( N%8 == 0); 
//     assert( K%8 == 0); 

//     size_t bytes_A = sizeof(float) * M * K;
//     size_t bytes_B = sizeof(float) * K * N;
//     size_t bytes_C = sizeof(float) * M * N;
//     float* h_A = (float*)malloc(bytes_A);
//     float* h_B = (float*)malloc(bytes_B);
//     float* h_C = (float*)malloc(bytes_C);
//     float* h_C1 = (float*)malloc(bytes_C);

//     float* d_A;
//     float* d_B;
//     float* d_C;

//     checkCudaErrors(cudaMalloc(&d_A, bytes_A));
//     checkCudaErrors(cudaMalloc(&d_B, bytes_B));
//     checkCudaErrors(cudaMalloc(&d_C, bytes_C));
//     double msecPerMatrixMul[2] = {0, 0};
//     double gigaFlops[2] = {0, 0};
//     double flopsPerMatrixMul = 2.0 * M * N * K;

//     const int BLOCK_SIZE_M = 128;
//     const int BLOCK_SIZE_K = 8;
//     const int BLOCK_SIZE_N = 128;
//     const int THREAD_SIZE_X = 8;
//     const int THREAD_SIZE_Y = 8;
//     const bool ENABLE_DOUBLE_BUFFER = true;

//     // generate A
//     for( int i = 0; i < M * K; i++ ){
//         h_A[i] = i / 13;
//     }

//     // generate B
//     for( int i = 0; i < K * N; i++ ) {
//         h_B[i] = i % 13;
//     }

//     checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
//     checkCudaErrors(cudaMemcpy( d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    
//     cudaEvent_t start, stop;
//     checkCudaErrors(cudaEventCreate(&start));
//     checkCudaErrors(cudaEventCreate(&stop));
//     float msecTotal = 0;
//     int nIter = 1000;

//     checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
//     checkCudaErrors(cudaEventRecord(start));
//     for (int run = 0 ; run < nIter; run ++ ) {
//         dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
//         dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
//         Sgemm<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> 
//         <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, M, K, N);
//     }
//     checkCudaErrors(cudaEventRecord(stop));
//     checkCudaErrors(cudaEventSynchronize(stop));
//     checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));


//     checkCudaErrors(cudaMemcpy( h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

//     msecPerMatrixMul[0] = msecTotal / nIter;
//     gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
//     printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
//         gigaFlops[0],
//         msecPerMatrixMul[0],
//         flopsPerMatrixMul);

//     // cublas
//     cublasHandle_t blas_handle;  
//     cublasCreate(&blas_handle);
//     float alpha = 1.0;
//     float beta = 0;
//     checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
//     checkCudaErrors(cudaEventRecord(start));
//     for (int run = 0 ; run < nIter; run ++ ) {
//         cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
//             M, N, K, &alpha, 
//             d_A, K, d_B, N, &beta, d_C, N
//         );
//     }
//     checkCudaErrors(cudaEventRecord(stop));
//     checkCudaErrors(cudaEventSynchronize(stop));
//     checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

//     checkCudaErrors(cudaMemcpy( h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

//     msecPerMatrixMul[1] = msecTotal / nIter;
//     gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
//     printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
//         gigaFlops[1],
//         msecPerMatrixMul[1],
//         flopsPerMatrixMul);

//     cublasDestroy(blas_handle); 
    
//     double eps = 1.e-6;  // machine zero
//     bool correct = true;
//     for (int i = 0; i < M * N; i++) {
//         int row = i / N;
//         int col = i % N;
//         double abs_err = fabs(h_C[i] - h_C1[col * M + row]);
//         double dot_length = M;
//         double abs_val = fabs(h_C[i]);
//         double rel_err = abs_err / abs_val / dot_length;
//         if (rel_err > eps) {
//             printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
//                     i, h_C[i], h_C1[col * M + row], eps);
//             correct = false;
//             break;
//         }
//     }

//     printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
//     printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);
    
//     // Free Memory
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);
    
//     free(h_A);
//     free(h_B);
//     free(h_C);
//     free(h_C1);
// }