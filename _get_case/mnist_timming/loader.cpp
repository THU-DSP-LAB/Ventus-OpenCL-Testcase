// loader_ptx_newscale.cpp — Driver API: launch "conv" 3 times with new sizes
// build (任选其一):
//   nvcc -O2 -std=c++17 loader_ptx_newscale.cpp -o loader_ptx -lcuda
//   g++  -O2 -std=c++17 -I/usr/local/cuda/include loader_ptx_newscale.cpp \
//        -L/usr/local/cuda/lib64 -lcuda -ldl -o loader_ptx
// run:
//   ./loader_ptx [kernel.ptx|kernel_sm89.cubin]

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <fstream>
#include <sstream>

#define CHECK_CUDA(call) do{ \
  CUresult _e=(call); \
  if(_e!=CUDA_SUCCESS){ const char* s=nullptr; cuGetErrorString(_e,&s); \
    std::fprintf(stderr,"CUDA error %d: %s @ %s:%d\n",(int)_e,s?s:"<null>",__FILE__,__LINE__); \
    std::exit(1); } \
}while(0)

/********** 与 OpenCL 侧一致的模型尺寸（新的 case） **********/
#define IN1_CHANNELS 1
#define IN1_H 28
#define IN1_W 28

#define CONV1_OUT_CHANNELS 16
#define CONV1_K 5
#define CONV1_OUT_H (IN1_H - CONV1_K + 1)   // 24
#define CONV1_OUT_W (IN1_W - CONV1_K + 1)   // 24

#define CONV2_OUT_CHANNELS 32
#define CONV2_K 5
#define CONV2_IN_CHANNELS CONV1_OUT_CHANNELS
#define CONV2_IN_H CONV1_OUT_H              // 24
#define CONV2_IN_W CONV1_OUT_W              // 24
#define CONV2_OUT_H (CONV2_IN_H - CONV2_K + 1)  // 20
#define CONV2_OUT_W (CONV2_IN_W - CONV2_K + 1)  // 20

#define CONV3_OUT_CHANNELS 10
#define CONV3_K 20
#define CONV3_IN_CHANNELS CONV2_OUT_CHANNELS
#define CONV3_IN_H CONV2_OUT_H              // 20
#define CONV3_IN_W CONV2_OUT_W              // 20
// conv3 输出 1x1

static std::string read_text_file(const char* path){
    std::ifstream ifs(path, std::ios::in|std::ios::binary);
    if(!ifs){ std::fprintf(stderr,"[ERR] cannot open %s\n",path); std::exit(1); }
    std::ostringstream ss; ss<<ifs.rdbuf(); return ss.str();
}

static CUmodule load_module_auto(const char* path){
    std::string spath(path);
    CUmodule mod=nullptr;
    if(spath.size()>=4 && spath.rfind(".ptx")==spath.size()-4){
        std::string ptx = read_text_file(path);
        CUjit_option opts[3];
        void*        vals[3];
        static char  logbuf[1<<16]; size_t logsz=sizeof(logbuf);
        opts[0]=CU_JIT_INFO_LOG_BUFFER;            vals[0]=logbuf;
        opts[1]=CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES; vals[1]=(void*)logsz;
        opts[2]=CU_JIT_TARGET_FROM_CUCONTEXT;      vals[2]=nullptr;
        CHECK_CUDA(cuModuleLoadDataEx(&mod, ptx.c_str(), 3, opts, vals));
        if(logbuf[0]) std::fprintf(stderr,"[JIT] %s\n",logbuf);
    }else{
        CHECK_CUDA(cuModuleLoad(&mod, path)); // cubin/fatbin
    }
    return mod;
}

int main(int argc, char** argv){
    const char* modfile = (argc>1? argv[1] : "kernel_NVIDIA_GeForce_RTX_4090.ptx");

    CHECK_CUDA(cuInit(0));
    CUdevice dev=0; CHECK_CUDA(cuDeviceGet(&dev,0));
    CUcontext ctx=nullptr; CHECK_CUDA(cuCtxCreate(&ctx,0,dev));

    CUmodule mod = load_module_auto(modfile);
    CUfunction fn=nullptr; CHECK_CUDA(cuModuleGetFunction(&fn, mod, "conv"));
    std::fprintf(stderr,"[INFO] loaded module: %s\n", modfile);

    // -------- 分配缓冲（零填充即可） --------
    // conv1
    size_t sz_in1  = (size_t)IN1_CHANNELS * IN1_H * IN1_W * sizeof(float);
    size_t sz_w1   = (size_t)CONV1_OUT_CHANNELS * IN1_CHANNELS * CONV1_K * CONV1_K * sizeof(float);
    size_t sz_b1   = (size_t)CONV1_OUT_CHANNELS * sizeof(float);
    size_t sz_o1   = (size_t)CONV1_OUT_CHANNELS * CONV1_OUT_H * CONV1_OUT_W * sizeof(float);
    // conv2
    size_t sz_w2   = (size_t)CONV2_OUT_CHANNELS * CONV2_IN_CHANNELS * CONV2_K * CONV2_K * sizeof(float);
    size_t sz_b2   = (size_t)CONV2_OUT_CHANNELS * sizeof(float);
    size_t sz_o2   = (size_t)CONV2_OUT_CHANNELS * CONV2_OUT_H * CONV2_OUT_W * sizeof(float);
    // conv3
    size_t sz_w3   = (size_t)CONV3_OUT_CHANNELS * CONV3_IN_CHANNELS * CONV3_K * CONV3_K * sizeof(float);
    size_t sz_b3   = (size_t)CONV3_OUT_CHANNELS * sizeof(float);
    size_t sz_o3   = (size_t)CONV3_OUT_CHANNELS * sizeof(float);

    CUdeviceptr d_in1=0,d_w1=0,d_b1=0,d_o1=0;
    CUdeviceptr d_w2=0,d_b2=0,d_o2=0;
    CUdeviceptr d_w3=0,d_b3=0,d_o3=0;

    CHECK_CUDA(cuMemAlloc(&d_in1, sz_in1));
    CHECK_CUDA(cuMemAlloc(&d_w1,  sz_w1));
    CHECK_CUDA(cuMemAlloc(&d_b1,  sz_b1));
    CHECK_CUDA(cuMemAlloc(&d_o1,  sz_o1));
    CHECK_CUDA(cuMemAlloc(&d_w2,  sz_w2));
    CHECK_CUDA(cuMemAlloc(&d_b2,  sz_b2));
    CHECK_CUDA(cuMemAlloc(&d_o2,  sz_o2));
    CHECK_CUDA(cuMemAlloc(&d_w3,  sz_w3));
    CHECK_CUDA(cuMemAlloc(&d_b3,  sz_b3));
    CHECK_CUDA(cuMemAlloc(&d_o3,  sz_o3));

    CHECK_CUDA(cuMemsetD8(d_in1,0,sz_in1));
    CHECK_CUDA(cuMemsetD8(d_w1, 0,sz_w1));  CHECK_CUDA(cuMemsetD8(d_b1,0,sz_b1)); CHECK_CUDA(cuMemsetD8(d_o1,0,sz_o1));
    CHECK_CUDA(cuMemsetD8(d_w2, 0,sz_w2));  CHECK_CUDA(cuMemsetD8(d_b2,0,sz_b2)); CHECK_CUDA(cuMemsetD8(d_o2,0,sz_o2));
    CHECK_CUDA(cuMemsetD8(d_w3, 0,sz_w3));  CHECK_CUDA(cuMemsetD8(d_b3,0,sz_b3)); CHECK_CUDA(cuMemsetD8(d_o3,0,sz_o3));

    auto launch = [&](CUdeviceptr in, CUdeviceptr w, CUdeviceptr b, CUdeviceptr out,
                      int ic,int ih,int iw,int kh,int kw,int oh,int ow,int relu,
                      unsigned gx,unsigned gy,unsigned gz,
                      unsigned bx,unsigned by,unsigned bz)
    {
        void* args[12] = {
            (void*)&in, (void*)&w, (void*)&b, (void*)&out,
            (void*)&ic, (void*)&ih, (void*)&iw,
            (void*)&kh, (void*)&kw,
            (void*)&oh, (void*)&ow,
            (void*)&relu
        };
        std::printf("[LAUNCH] grid=(%u,%u,%u) block=(%u,%u,%u)\n",gx,gy,gz,bx,by,bz);
        CHECK_CUDA(cuLaunchKernel(fn, gx,gy,gz,  bx,by,bz,  0,0, args, nullptr));
    };

    // ===================== Launch 3 层（新规模） =====================

    // layer1: num_warp=1, num_thread=32, num_wg=512, kernel_size=(8,8,8)
    launch(
        d_in1, d_w1, d_b1, d_o1,
        IN1_CHANNELS, IN1_H, IN1_W,
        CONV1_K, CONV1_K,
        CONV1_OUT_H, CONV1_OUT_W,
        /*relu*/1,
        /*grid*/ 8, 8, 8,
        /*block*/32, 1, 1
    );
    CHECK_CUDA(cuCtxSynchronize());

    // layer2: num_warp=1, num_thread=32, num_wg=512, kernel_size=(32,4,4)
    launch(
        /*in =*/ d_o1, /*w=*/ d_w2, /*b=*/ d_b2, /*out=*/ d_o2,
        CONV2_IN_CHANNELS, CONV2_IN_H, CONV2_IN_W,
        CONV2_K, CONV2_K,
        CONV2_OUT_H, CONV2_OUT_W,
        /*relu*/1,
        /*grid*/ 32, 4, 4,
        /*block*/32, 1, 1
    );
    CHECK_CUDA(cuCtxSynchronize());

    // layer3: num_warp=1, num_thread=32, num_wg=1, kernel_size=(1,1,1)
    launch(
        /*in =*/ d_o2, /*w=*/ d_w3, /*b=*/ d_b3, /*out=*/ d_o3,
        CONV3_IN_CHANNELS, CONV3_IN_H, CONV3_IN_W,
        CONV3_K, CONV3_K,
        /*oh*/1, /*ow*/1,
        /*relu*/0,
        /*grid*/ 1, 1, 1,
        /*block*/32, 1, 1
    );
    CHECK_CUDA(cuCtxSynchronize());

    std::fprintf(stderr,"[loader_ptx] all conv launches done (module=%s)\n", modfile);

    // 资源释放
    cuMemFree(d_in1);
    cuMemFree(d_w1); cuMemFree(d_b1); cuMemFree(d_o1);
    cuMemFree(d_w2); cuMemFree(d_b2); cuMemFree(d_o2);
    cuMemFree(d_w3); cuMemFree(d_b3); cuMemFree(d_o3);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    return 0;
}
