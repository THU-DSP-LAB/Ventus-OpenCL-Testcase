// launch_ptx_sgemm.cpp — 用 CUDA Driver API 加载 OpenCL 导出的 PTX 并发射 sgemm_one_wg
// 构建：nvcc -std=c++17 launch_ptx_sgemm.cpp -lcuda -o launch_ptx_sgemm
// 运行：./launch_ptx_sgemm kernel.ptx 32 32 32   # 可选参数：PTX路径 M N K
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <cassert>

#define CHECK_CUDA(call) do{                                \
    CUresult _e = (call);                                   \
    if(_e != CUDA_SUCCESS){                                 \
        const char* _err = nullptr;                         \
        cuGetErrorString(_e, &_err);                        \
        fprintf(stderr,"CUDA error %d: %s at %s:%d\n",      \
                (int)_e, _err?_err:"<unknown>", __FILE__, __LINE__); \
        std::exit(1);                                       \
    }                                                       \
}while(0)

// === 必须与 PTX 编译时一致 ===
// 你当前导出的 PTX 用的是 TILE_M=1, TILE_N=1；若以 16×16 重新编译 PTX，请改为 16,16。
#ifndef TILE_M
#define TILE_M 1
#endif
#ifndef TILE_N
#define TILE_N 1
#endif

static void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            float acc = 0.0f;
            for(int k=0;k<K;k++) acc += A[i*K + k] * B[k*N + j];
            C[i*N + j] = acc;
        }
    }
}

static void fill_data(std::vector<float>& v, float s1, float s2){
    for(size_t i=0;i<v.size();++i){
        v[i] = 0.5f*sinf((float)i*s1) + 0.5f*cosf((float)i*s2);
    }
}

// 如果你不确定 PTX 内的入口名，可以先用文本编辑器或 `grep ".entry"` 看 .entry 名称
static const char* KERNEL_NAME = "sgemm_one_wg";

int main(int argc, char** argv){
    // 参数：ptx路径 M N K（可选）
    std::string ptx_path = "kernel.ptx";
    int M = 128/4, N = 128/4, K = 128/4;
    if(argc >= 2) ptx_path = argv[1];
    if(argc >= 5){ M = std::atoi(argv[2]); N = std::atoi(argv[3]); K = std::atoi(argv[4]); }

    printf("[INFO] Using PTX: %s\n", ptx_path.c_str());
    printf("[INFO] M=%d N=%d K=%d | TILE_M=%d TILE_N=%d (grid=1x1x1, block=%dx%dx1)\n",
           M,N,K, TILE_M, TILE_N, TILE_N, TILE_M);

    // 1) 初始化 Driver API
    CHECK_CUDA(cuInit(0));
    CUdevice dev; CHECK_CUDA(cuDeviceGet(&dev, 0));
    char name[200]; cuDeviceGetName(name, sizeof(name), dev);
    printf("[INFO] Device: %s\n", name);

    CUcontext ctx; CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

    // 2) 加载 PTX 模块 & 获取函数
    CUmodule mod; CHECK_CUDA(cuModuleLoad(&mod, ptx_path.c_str()));
    CUfunction fun; CHECK_CUDA(cuModuleGetFunction(&fun, mod, KERNEL_NAME));

    // 3) 准备数据
    size_t szA = (size_t)M * K;
    size_t szB = (size_t)K * N;
    size_t szC = (size_t)M * N;

    std::vector<float> hA(szA), hB(szB), hC(szC, 0.0f), hRef(szC, 0.0f);
    fill_data(hA, 0.01f, 0.013f);
    fill_data(hB, 0.02f, 0.017f);
    cpu_gemm(hA.data(), hB.data(), hRef.data(), M,N,K);

    CUdeviceptr dA, dB, dC;
    CHECK_CUDA(cuMemAlloc(&dA, szA * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&dB, szB * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&dC, szC * sizeof(float)));
    CHECK_CUDA(cuMemcpyHtoD(dA, hA.data(), szA * sizeof(float)));
    CHECK_CUDA(cuMemcpyHtoD(dB, hB.data(), szB * sizeof(float)));

    // 4) 设置 kernel 参数（与 OpenCL 一样的顺序与类型）
    void* args[] = {
        &dA, &dB, &dC, &M, &N, &K
    };

    // 5) 计时事件
    CUevent evStart, evStop;
    CHECK_CUDA(cuEventCreate(&evStart, 0));
    CHECK_CUDA(cuEventCreate(&evStop, 0));

    // 6) 启动：仅 1 个 block，形状 (TILE_N, TILE_M, 1)
    //    内核内部会通过 (bm,bn,bk) 循环覆盖整张 C。
    CHECK_CUDA(cuEventRecord(evStart, 0));
    CHECK_CUDA(cuLaunchKernel(
        fun,
        /* grid */ 1, 1, 1,
        /* block*/ TILE_N, TILE_M, 1,
        /* sharedMemBytes */ 0,
        /* stream */ 0,
        args,
        /* extra */ nullptr
    ));
    CHECK_CUDA(cuEventRecord(evStop, 0));
    CHECK_CUDA(cuEventSynchronize(evStop));

    // 7) 读回 & 校验
    CHECK_CUDA(cuMemcpyDtoH(hC.data(), dC, szC * sizeof(float)));

    float max_abs_rel = 0.0f;
    int bad = 0;
    for(size_t i=0;i<szC;i++){
        float ref = hRef[i];
        float rel = (ref==0.0f) ? std::fabs(hC[i]) : std::fabs((hC[i]-ref)/ref);
        if(rel > max_abs_rel) max_abs_rel = rel;
        if(rel > 1e-3f) bad++;
    }

    float ms = 0.0f;
    CHECK_CUDA(cuEventElapsedTime(&ms, evStart, evStop)); // ms
    double gflops = (2.0 * (double)M * (double)N * (double)K) / (ms * 1.0e6);

    printf("\n=== SGEMM (PTX via Driver API) ===\n");
    printf("time: %.3f ms | perf: %.2f GFLOP/s | max rel err: %.3e | bad: %d/%zu\n",
           ms, gflops, max_abs_rel, bad, szC);

    // 8) 资源释放
    cuEventDestroy(evStart); cuEventDestroy(evStop);
    cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    return 0;
}
