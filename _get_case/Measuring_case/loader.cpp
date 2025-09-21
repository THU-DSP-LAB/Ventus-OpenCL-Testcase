// loader.cpp  -- minimal CUDA Driver API launcher for OpenCL-generated cubin
// build: g++ -O2 loader.cpp -o loader -ldl -lcuda
// run:   ncu --target-processes all --set full \
//          --metrics sm__inst_executed.avg.per_cycle_active,sm__inst_executed.sum,sm__cycles_active.sum \
//          ./loader

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CHECK_CUDA(call) do {                             \
    CUresult _e=(call);                                   \
    if(_e!=CUDA_SUCCESS){                                 \
        const char* s=nullptr; cuGetErrorString(_e,&s);   \
        fprintf(stderr,"CUDA error %d: %s @ %s:%d\n",     \
                (int)_e, s?s:"<null>", __FILE__, __LINE__); \
        exit(1);                                          \
    }                                                     \
} while(0)

/* ---- 从你的 main.c 复制来的网络超参（conv1 这一层） ---- */
#define IN1_CHANNELS 1
#define IN1_H 28
#define IN1_W 28
#define CONV1_OUT_CHANNELS 2
#define CONV1_K 5
#define CONV1_STRIDE 1
#define CONV1_OUT_H (IN1_H - CONV1_K + 1)   /* 24 */
#define CONV1_OUT_W (IN1_W - CONV1_K + 1)   /* 24 */

int main(){
    /* 1) 上下文 */
    CHECK_CUDA(cuInit(0));
    CUdevice dev; CHECK_CUDA(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

    /* 2) 加载 cubin 模块 & 获取 kernel */
    CUmodule mod; CHECK_CUDA(cuModuleLoad(&mod, "kernel_sm89.cubin"));
    // 入口名与 PTX/.entry 一致；你的 OpenCL 里叫 "conv"
    CUfunction fn; CHECK_CUDA(cuModuleGetFunction(&fn, mod, "sgemm_one_wg"));

    /* 3) 为 conv1 分配设备内存（只要能跑即可，数据内容无所谓） */
    size_t sz_in  = (size_t)IN1_CHANNELS * IN1_H * IN1_W * sizeof(float);
    size_t sz_w   = (size_t)CONV1_OUT_CHANNELS * IN1_CHANNELS * CONV1_K * CONV1_K * sizeof(float);
    size_t sz_b   = (size_t)CONV1_OUT_CHANNELS * sizeof(float);
    size_t sz_out = (size_t)CONV1_OUT_CHANNELS * CONV1_OUT_H * CONV1_OUT_W * sizeof(float);

    CUdeviceptr d_in=0, d_w=0, d_b=0, d_out=0;
    CHECK_CUDA(cuMemAlloc(&d_in,  sz_in));
    CHECK_CUDA(cuMemAlloc(&d_w,   sz_w));
    CHECK_CUDA(cuMemAlloc(&d_b,   sz_b));
    CHECK_CUDA(cuMemAlloc(&d_out, sz_out));
    CHECK_CUDA(cuMemsetD8(d_in,  0, sz_in));
    CHECK_CUDA(cuMemsetD8(d_w,   0, sz_w));
    CHECK_CUDA(cuMemsetD8(d_b,   0, sz_b));
    CHECK_CUDA(cuMemsetD8(d_out, 0, sz_out));

    /* 4) 准备参数（严格按 OpenCL 里的顺序与类型） */
    int ic = IN1_CHANNELS, ih = IN1_H, iw = IN1_W;
    int kh = CONV1_K, kw = CONV1_K;
    int oh = CONV1_OUT_H, ow = CONV1_OUT_W;
    int relu = 1;
    int sh = CONV1_STRIDE, sw = CONV1_STRIDE;

    void* args[] = {
        (void*)&d_in,
        (void*)&d_w,
        (void*)&d_b,
        (void*)&d_out,
        (void*)&ic, (void*)&ih, (void*)&iw,
        (void*)&kh, (void*)&kw,
        (void*)&oh, (void*)&ow,
        (void*)&relu,
        (void*)&sh, (void*)&sw
    };

    /* 如果你的 PTX .entry 多了 OpenCL 隐式参数（global_offset[3] 等），
       打开 kernel.ptx 确认后可在此处追加：
       int go0=0, go1=0, go2=0;
       void* args[] = { ... , &sh, &sw, &go0, &go1, &go2 };
       并确保顺序与 .entry 一致。 */

    /* 5) 发射一次：grid = (OC, OH, OW), block = (1,1,1) */
    unsigned gx = CONV1_OUT_CHANNELS;
    unsigned gy = CONV1_OUT_H;
    unsigned gz = CONV1_OUT_W;
    CHECK_CUDA(cuLaunchKernel(
        fn,
        gx, gy, gz,     // gridDim
        1, 1, 1,        // blockDim
        0,              // sharedMemBytes
        0,              // stream
        args,           // kernel params
        nullptr         // extra
    ));
    CHECK_CUDA(cuCtxSynchronize());

    /* 6) 清理 */
    cuMemFree(d_in); cuMemFree(d_w); cuMemFree(d_b); cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);

    fprintf(stderr, "[loader] launch done.\n");
    return 0;
}
