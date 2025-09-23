// loader_mnist_small_fixed.cpp — Driver API, no dim3; launch 3 times with given sizes
// build (Driver API only):
//   g++ -O2 -std=c++17 loader_mnist_small_fixed.cpp -o loader -lcuda -ldl
// or with nvcc:
//   nvcc -O2 -std=c++17 loader_mnist_small_fixed.cpp -o loader -lcuda
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CHECK_CUDA(call) do{ \
  CUresult _e=(call); \
  if(_e!=CUDA_SUCCESS){ const char* s=nullptr; cuGetErrorString(_e,&s); \
    std::fprintf(stderr,"CUDA error %d: %s @ %s:%d\n",(int)_e,s?s:"<null>",__FILE__,__LINE__); \
    std::exit(1); } \
}while(0)

/* ---- 与 OpenCL main.c 一致的模型超参 ---- */
#define IN1_CHANNELS 1
#define IN1_H 28
#define IN1_W 28
#define CONV1_OUT_CHANNELS 2
#define CONV1_K 5
#define CONV1_STRIDE 1
#define CONV1_OUT_H (IN1_H - CONV1_K + 1)   /* 24 */
#define CONV1_OUT_W (IN1_W - CONV1_K + 1)   /* 24 */

#define CONV2_OUT_CHANNELS 1
#define CONV2_K 5
#define CONV2_STRIDE 5
#define CONV2_IN_CHANNELS  CONV1_OUT_CHANNELS
#define CONV2_IN_H         CONV1_OUT_H
#define CONV2_IN_W         CONV1_OUT_W
#define CONV2_OUT_H ((CONV2_IN_H - CONV2_K) / CONV2_STRIDE + 1) /* 4 */
#define CONV2_OUT_W ((CONV2_IN_W - CONV2_K) / CONV2_STRIDE + 1) /* 4 */

#define CONV3_OUT_CHANNELS 10
#define CONV3_K 4
#define CONV3_STRIDE 1
#define CONV3_IN_CHANNELS  CONV2_OUT_CHANNELS
#define CONV3_IN_H         CONV2_OUT_H
#define CONV3_IN_W         CONV2_OUT_W
#define CONV3_OUT_H 1
#define CONV3_OUT_W 1

static CUmodule loadModuleAuto(const char** used){
    CUmodule mod;
    if(used) *used = "kernel_sm89.cubin";
    if(cuModuleLoad(&mod, "kernel_sm89.cubin") == CUDA_SUCCESS) return mod;
    if(used) *used = "kernel.ptx";
    CHECK_CUDA(cuModuleLoad(&mod, "kernel.ptx"));
    return mod;
}

int main(){
    CHECK_CUDA(cuInit(0));
    CUdevice dev; CHECK_CUDA(cuDeviceGet(&dev,0));
    CUcontext ctx; CHECK_CUDA(cuCtxCreate(&ctx,0,dev));

    const char* used=nullptr;
    CUmodule mod = loadModuleAuto(&used);
    std::fprintf(stderr,"[INFO] loaded module: %s\n", used);
    CUfunction fn; CHECK_CUDA(cuModuleGetFunction(&fn, mod, "conv"));

    // --- 设备内存（置零即可） ---
    size_t sz_in  = (size_t)IN1_CHANNELS * IN1_H * IN1_W * sizeof(float);
    size_t sz_w1  = (size_t)CONV1_OUT_CHANNELS * IN1_CHANNELS * CONV1_K * CONV1_K * sizeof(float);
    size_t sz_b1  = (size_t)CONV1_OUT_CHANNELS * sizeof(float);
    size_t sz_o1  = (size_t)CONV1_OUT_CHANNELS * CONV1_OUT_H * CONV1_OUT_W * sizeof(float);

    size_t sz_w2  = (size_t)CONV2_OUT_CHANNELS * CONV2_IN_CHANNELS * CONV2_K * CONV2_K * sizeof(float);
    size_t sz_b2  = (size_t)CONV2_OUT_CHANNELS * sizeof(float);
    size_t sz_o2  = (size_t)CONV2_OUT_CHANNELS * CONV2_OUT_H * CONV2_OUT_W * sizeof(float);

    size_t sz_w3  = (size_t)CONV3_OUT_CHANNELS * CONV3_IN_CHANNELS * CONV3_K * CONV3_K * sizeof(float);
    size_t sz_b3  = (size_t)CONV3_OUT_CHANNELS * sizeof(float);
    size_t sz_o3  = (size_t)CONV3_OUT_CHANNELS * sizeof(float);

    CUdeviceptr d_in, d_w1, d_b1, d_o1;
    CUdeviceptr d_w2, d_b2, d_o2;
    CUdeviceptr d_w3, d_b3, d_o3;

    CHECK_CUDA(cuMemAlloc(&d_in, sz_in));
    CHECK_CUDA(cuMemAlloc(&d_w1, sz_w1)); CHECK_CUDA(cuMemAlloc(&d_b1, sz_b1)); CHECK_CUDA(cuMemAlloc(&d_o1, sz_o1));
    CHECK_CUDA(cuMemAlloc(&d_w2, sz_w2)); CHECK_CUDA(cuMemAlloc(&d_b2, sz_b2)); CHECK_CUDA(cuMemAlloc(&d_o2, sz_o2));
    CHECK_CUDA(cuMemAlloc(&d_w3, sz_w3)); CHECK_CUDA(cuMemAlloc(&d_b3, sz_b3)); CHECK_CUDA(cuMemAlloc(&d_o3, sz_o3));

    CHECK_CUDA(cuMemsetD8(d_in, 0, sz_in));
    CHECK_CUDA(cuMemsetD8(d_w1, 0, sz_w1)); CHECK_CUDA(cuMemsetD8(d_b1, 0, sz_b1)); CHECK_CUDA(cuMemsetD8(d_o1, 0, sz_o1));
    CHECK_CUDA(cuMemsetD8(d_w2, 0, sz_w2)); CHECK_CUDA(cuMemsetD8(d_b2, 0, sz_b2)); CHECK_CUDA(cuMemsetD8(d_o2, 0, sz_o2));
    CHECK_CUDA(cuMemsetD8(d_w3, 0, sz_w3)); CHECK_CUDA(cuMemsetD8(d_b3, 0, sz_b3)); CHECK_CUDA(cuMemsetD8(d_o3, 0, sz_o3));

    // ---------------- layer1 ----------------
    {
        int ic=IN1_CHANNELS, ih=IN1_H, iw=IN1_W;
        int kh=CONV1_K, kw=CONV1_K;
        int oh=CONV1_OUT_H, ow=CONV1_OUT_W;
        int relu=1, sh=CONV1_STRIDE, sw=CONV1_STRIDE;

        void* args[] = { &d_in,&d_w1,&d_b1,&d_o1,
                         &ic,&ih,&iw,&kh,&kw,&oh,&ow,&relu,&sh,&sw };
        // 若 .entry 末尾还有 global_offset[3] 等隐式参数，请在此处追加 3 个 int 0，并保持顺序。

        unsigned gridX=1, gridY=1, gridZ=8;      // kernel_size (1,1,8)
        unsigned blockX=160, blockY=1, blockZ=1; // 5 warps * 32 threads
        std::printf("[LAUNCH] layer1 grid=(%u,%u,%u) block=(%u,%u,%u)\n",
                    gridX,gridY,gridZ, blockX,blockY,blockZ);
        CHECK_CUDA(cuLaunchKernel(fn,
            gridX,gridY,gridZ, blockX,blockY,blockZ,
            0, 0, args, nullptr));
        CHECK_CUDA(cuCtxSynchronize());
    }

    // ---------------- layer2 ----------------
    {
        int ic=CONV2_IN_CHANNELS, ih=CONV2_IN_H, iw=CONV2_IN_W;
        int kh=CONV2_K, kw=CONV2_K;
        int oh=CONV2_OUT_H, ow=CONV2_OUT_W;
        int relu=1, sh=CONV2_STRIDE, sw=CONV2_STRIDE;

        CUdeviceptr d_in2 = d_o1;
        void* args[] = { &d_in2,&d_w2,&d_b2,&d_o2,
                         &ic,&ih,&iw,&kh,&kw,&oh,&ow,&relu,&sh,&sw };

        unsigned gridX=1, gridY=1, gridZ=1;     // kernel_size (1,1,1)
        unsigned blockX=32, blockY=1, blockZ=1; // 1 warp * 32 threads
        std::printf("[LAUNCH] layer2 grid=(%u,%u,%u) block=(%u,%u,%u)\n",
                    gridX,gridY,gridZ, blockX,blockY,blockZ);
        CHECK_CUDA(cuLaunchKernel(fn,
            gridX,gridY,gridZ, blockX,blockY,blockZ,
            0, 0, args, nullptr));
        CHECK_CUDA(cuCtxSynchronize());
    }

    // ---------------- layer3 ----------------
    {
        int ic=CONV3_IN_CHANNELS, ih=CONV3_IN_H, iw=CONV3_IN_W;
        int kh=CONV3_K, kw=CONV3_K;
        int oh=CONV3_OUT_H, ow=CONV3_OUT_W;
        int relu=0, sh=CONV3_STRIDE, sw=CONV3_STRIDE;

        CUdeviceptr d_in3 = d_o2;
        void* args[] = { &d_in3,&d_w3,&d_b3,&d_o3,
                         &ic,&ih,&iw,&kh,&kw,&oh,&ow,&relu,&sh,&sw };

        unsigned gridX=1, gridY=1, gridZ=1;     // kernel_size (1,1,1)
        unsigned blockX=32, blockY=1, blockZ=1; // 1 warp * 32 threads
        std::printf("[LAUNCH] layer3 grid=(%u,%u,%u) block=(%u,%u,%u)\n",
                    gridX,gridY,gridZ, blockX,blockY,blockZ);
        CHECK_CUDA(cuLaunchKernel(fn,
            gridX,gridY,gridZ, blockX,blockY,blockZ,
            0, 0, args, nullptr));
        CHECK_CUDA(cuCtxSynchronize());
    }

    // 清理
    cuMemFree(d_in);
    cuMemFree(d_w1); cuMemFree(d_b1); cuMemFree(d_o1);
    cuMemFree(d_w2); cuMemFree(d_b2); cuMemFree(d_o2);
    cuMemFree(d_w3); cuMemFree(d_b3); cuMemFree(d_o3);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    std::fprintf(stderr,"[INFO] all 3 launches done.\n");
    return 0;
}
