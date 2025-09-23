// main.cu — 1024x1024x1024 GEMM (C = A * B), tiled shared memory
// build: nvcc -O3 -std=c++17 -arch=sm_89 -lineinfo -Xptxas -O3 -o gemm main.cu
// run:   ./gemm
// ncu:   ncu --kernel-name-base demangled --kernel-name ::gemm_tiled \
//           --set full --metrics sm__inst_executed.avg.per_cycle_active,sm__inst_executed.sum,sm__cycles_active.sum ./gemm

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#ifndef N
#define N 1024
#endif

#ifndef TILE
#define TILE 32
#endif

#define CUDA_CHECK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", _e, cudaGetErrorString(_e), __FILE__, __LINE__); \
    std::exit(1); \
  } \
} while(0)

__global__ void gemm_tiled(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int n)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    #pragma unroll
    for (int t = 0; t < n; t += TILE) {
        // 每个线程各搬一个元素到共享内存（对齐、合并良好）
        As[threadIdx.y][threadIdx.x] = A[row * n + (t + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * n + col];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc = fmaf(As[threadIdx.y][k], Bs[k][threadIdx.x], acc);

        __syncthreads();
    }
    C[row * n + col] = acc;
}

int main(){
    const int n = N;
    const size_t bytes = size_t(n) * n * sizeof(float);

    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);

    // 简单初始化（避免 Host 计算校验的大开销）
    for (int i = 0; i < n*n; ++i) {
        hA[i] = (i % 3) * 0.5f;
        hB[i] = (i % 5) * 0.25f;
    }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid(n / TILE, n / TILE);

    // 预热
    for (int i = 0; i < 3; ++i) {
        gemm_tiled<<<grid, block>>>(dA, dB, dC, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    gemm_tiled<<<grid, block>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    // 理论 FLOPs: 2 * N^3（乘加算两次）
    double gflops = (2.0 * n * n * n) / (ms * 1e6);
    printf("GEMM %dx%dx%d | time = %.3f ms | %.2f GFLOP/s\n", n, n, n, ms, gflops);
    printf("launch: grid=(%d,%d) block=(%d,%d)\n", grid.x, grid.y, block.x, block.y);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
