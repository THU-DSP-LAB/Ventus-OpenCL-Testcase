// resnet18_main.cc — Host side to run ResNet-18 forward with OpenCL kernels (deterministic & safe releases)
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cassert>
#include <random>

// ======= 路径（相对于本文件所在工作目录）=======
static const char* KPATH_CONV      = "../AIops/Conv/conv.cl";
static const char* KPATH_CONV_BN   = "../AIops/Conv_BN/conv_bn.cl";
static const char* KPATH_BN        = "../AIops/BatchNorm2d/bn2d.cl";
static const char* KPATH_RELU      = "../AIops/ReLU/relu.cl";
static const char* KPATH_ADD       = "../AIops/add/add.cl";
static const char* KPATH_POOL2D    = "../AIops/Pool2D/pool2d.cl";
static const char* KPATH_GEMM      = "../AIops/GEMM/gemm.cl";

// ======= 配置 =======
static const int  NUM_CLASSES = 1000;
static const int  N = 1;                // Batch 固定 1，布局 NCHW
static const int  IN_C = 3, IN_H = 224, IN_W = 224;
static const float BN_EPS = 1e-5f;
static const bool  USE_FUSED_CONV_BN = true; // 改为 false 可使用 Conv + BN 两步

// ======= 固定随机工具 =======
static inline uint32_t seed_mix(uint32_t base, uint32_t tag){
    uint32_t x = base ^ (tag + 0x9e3779b9u + (base<<6) + (base>>2));
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
static void fill_uniform(std::vector<float>& v, float lo, float hi, uint32_t seed){
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto &x : v) x = dist(rng);
}

// ======= I/O 与 OpenCL 工具 =======
static char* load_text_file(const char* path, size_t* out_size = nullptr){
    FILE* fp = fopen(path, "rb");
    if(!fp){ fprintf(stderr, "无法打开 kernel 文件: %s\n", path); exit(1); }
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); rewind(fp);
    char* buf = (char*)malloc(sz+1);
    fread(buf,1,sz,fp); buf[sz]='\0'; fclose(fp);
    if(out_size) *out_size = (size_t)sz;
    return buf;
}
static cl_program build_program(cl_context ctx, cl_device_id dev, const char* path){
    size_t sz=0; char* src = load_text_file(path, &sz);
    cl_int err=CL_SUCCESS;
    const char* p = src;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &p, &sz, &err);
    if(err!=CL_SUCCESS){ fprintf(stderr,"clCreateProgramWithSource failed: %d\n", err); exit(1); }
    err = clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
    if(err!=CL_SUCCESS){
        size_t log_sz=0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        std::vector<char> log(log_sz+1);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log.data(), NULL);
        log[log_sz]='\0';
        fprintf(stderr,"Build log for %s:\n%s\n", path, log.data());
        exit(1);
    }
    free(src);
    return prog;
}

static void check(cl_int e, const char* msg){ if(e!=CL_SUCCESS){ fprintf(stderr,"%s failed: %d\n", msg, e); exit(1);} }
static void zero_buffer(cl_command_queue q, cl_mem buf, size_t bytes){
    const cl_uint zero = 0; cl_int e = clEnqueueFillBuffer(q, buf, &zero, sizeof(zero), 0, bytes, 0, NULL, NULL);
    check(e, "FillBuffer(0)");
}
static void safe_release_mem(cl_mem &m){ if(m){ clReleaseMemObject(m); m=nullptr; } }

struct Tensor {
    int C=0,H=0,W=0;        // N=1 固定
    std::vector<float> host;
    cl_mem buf = nullptr;
    Tensor(){}
    Tensor(int c,int h,int w):C(c),H(h),W(w),host((size_t)c*h*w) {}
    size_t bytes() const { return (size_t)C*H*W*sizeof(float); }
    size_t elems() const { return (size_t)C*H*W; }
};

static cl_mem make_device_buffer(cl_context ctx, const void* host, size_t bytes, cl_int* err=nullptr){
    cl_int e;
    cl_mem b = clCreateBuffer(ctx, CL_MEM_READ_WRITE | (host?CL_MEM_COPY_HOST_PTR:0), bytes, (void*)host, &e);
    if(err) *err=e;
    check(e, "clCreateBuffer");
    return b;
}
static void upload_tensor(cl_command_queue q, cl_mem buf, const void* src, size_t bytes){
    cl_int e = clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, bytes, src, 0, NULL, NULL);
    check(e, "EnqueueWriteBuffer");
}
static void download_tensor(cl_command_queue q, cl_mem buf, void* dst, size_t bytes){
    cl_int e = clEnqueueReadBuffer(q, buf, CL_TRUE, 0, bytes, dst, 0, NULL, NULL);
    check(e, "EnqueueReadBuffer");
}

// ======= kernel 句柄集合 =======
struct Kernels {
    cl_program prog_conv=nullptr, prog_convbn=nullptr, prog_bn=nullptr;
    cl_program prog_relu=nullptr, prog_add=nullptr, prog_pool=nullptr, prog_gemm=nullptr;

    cl_kernel  k_conv=nullptr, k_convbn=nullptr, k_bn=nullptr;
    cl_kernel  k_relu=nullptr, k_add=nullptr, k_pool=nullptr, k_gemm=nullptr;
};
static void build_all_kernels(cl_context ctx, cl_device_id dev, Kernels& K){
    cl_int err;
    if (USE_FUSED_CONV_BN) {
        K.prog_convbn = build_program(ctx, dev, KPATH_CONV_BN);
        K.k_convbn    = clCreateKernel(K.prog_convbn, "conv2d_bn", &err); check(err,"clCreateKernel(conv2d_bn)");
    } else {
        K.prog_conv   = build_program(ctx, dev, KPATH_CONV);
        K.k_conv      = clCreateKernel(K.prog_conv, "conv2d", &err); check(err,"clCreateKernel(conv2d)");
        K.prog_bn     = build_program(ctx, dev, KPATH_BN);
        K.k_bn        = clCreateKernel(K.prog_bn, "batchnorm2d_infer", &err); check(err,"clCreateKernel(batchnorm2d_infer)");
    }
    K.prog_relu   = build_program(ctx, dev, KPATH_RELU);
    K.k_relu      = clCreateKernel(K.prog_relu, "relu3d", &err); check(err,"clCreateKernel(relu3d)");
    K.prog_add    = build_program(ctx, dev, KPATH_ADD);
    K.k_add       = clCreateKernel(K.prog_add, "add3d", &err);   check(err,"clCreateKernel(add3d)");
    K.prog_pool   = build_program(ctx, dev, KPATH_POOL2D);
    K.k_pool      = clCreateKernel(K.prog_pool, "pool2d", &err); check(err,"clCreateKernel(pool2d)");
    K.prog_gemm   = build_program(ctx, dev, KPATH_GEMM);
    K.k_gemm      = clCreateKernel(K.prog_gemm, "gemm", &err);   check(err,"clCreateKernel(gemm)");
}

// ======= 算子封装（host 调用）=======

// Conv_BN：  output[OC,OH,OW] = BN( Conv(input[IC,IH,IW]) )
static void run_conv_bn(cl_command_queue q, cl_kernel k,
                        cl_mem in, cl_mem w, cl_mem b, cl_mem gamma, cl_mem beta, cl_mem mean, cl_mem var,
                        float eps,
                        cl_mem out,
                        int IC,int IH,int IW, int OC,int KH,int KW, int OH,int OW, int SH,int SW, int PH,int PW)
{
    int arg=0; cl_int e=CL_SUCCESS;
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&in);
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&w);
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&b);
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&gamma);
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&beta);
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&mean);
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&var);
    e|=clSetKernelArg(k,arg++,sizeof(float),&eps);
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&out);
    e|=clSetKernelArg(k,arg++,sizeof(int),&IC);
    e|=clSetKernelArg(k,arg++,sizeof(int),&IH);
    e|=clSetKernelArg(k,arg++,sizeof(int),&IW);
    e|=clSetKernelArg(k,arg++,sizeof(int),&OC);
    e|=clSetKernelArg(k,arg++,sizeof(int),&KH);
    e|=clSetKernelArg(k,arg++,sizeof(int),&KW);
    e|=clSetKernelArg(k,arg++,sizeof(int),&OH);
    e|=clSetKernelArg(k,arg++,sizeof(int),&OW);
    e|=clSetKernelArg(k,arg++,sizeof(int),&SH);
    e|=clSetKernelArg(k,arg++,sizeof(int),&SW);
    e|=clSetKernelArg(k,arg++,sizeof(int),&PH);
    e|=clSetKernelArg(k,arg++,sizeof(int),&PW);
    check(e,"set conv_bn args");

    size_t g[3]={(size_t)OC,(size_t)OH,(size_t)OW};
    e = clEnqueueNDRangeKernel(q, k, 3, NULL, g, NULL, 0, NULL, NULL);
    check(e,"enqueue conv_bn");
}

// BN（推理）
static void run_bn_infer(cl_command_queue q, cl_kernel k,
                         cl_mem x, cl_mem gamma, cl_mem beta, cl_mem mean, cl_mem var, float eps,
                         cl_mem y, int C,int H,int W)
{
    int arg=0; cl_int e=CL_SUCCESS;
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&x);
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&gamma);
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&beta);
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&mean);
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&var);
    e|=clSetKernelArg(k,arg++,sizeof(float),&eps);
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&y);
    e|=clSetKernelArg(k,arg++,sizeof(int),&C);
    e|=clSetKernelArg(k,arg++,sizeof(int),&H);
    e|=clSetKernelArg(k,arg++,sizeof(int),&W);
    check(e,"set bn args");
    size_t g[3]={(size_t)C,(size_t)H,(size_t)W};
    e = clEnqueueNDRangeKernel(q, k, 3, NULL, g, NULL, 0, NULL, NULL);
    check(e,"enqueue bn");
}

// ReLU
static void run_relu(cl_command_queue q, cl_kernel k, cl_mem x, cl_mem y, int C,int H,int W){
    int arg=0; cl_int e=CL_SUCCESS;
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&x);
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&y);
    e|=clSetKernelArg(k,arg++,sizeof(int),&C);
    e|=clSetKernelArg(k,arg++,sizeof(int),&H);
    e|=clSetKernelArg(k,arg++,sizeof(int),&W);
    check(e,"set relu args");
    size_t g[3]={(size_t)C,(size_t)H,(size_t)W};
    e = clEnqueueNDRangeKernel(q, k, 3, NULL, g, NULL, 0, NULL, NULL);
    check(e,"enqueue relu");
}

// Add
static void run_add(cl_command_queue q, cl_kernel k, cl_mem a, cl_mem b, cl_mem out, int C,int H,int W){
    int arg=0; cl_int e=CL_SUCCESS;
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&a);
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&b);
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&out);
    e|=clSetKernelArg(k,arg++,sizeof(int),&C);
    e|=clSetKernelArg(k,arg++,sizeof(int),&H);
    e|=clSetKernelArg(k,arg++,sizeof(int),&W);
    check(e,"set add args");
    size_t g[3]={(size_t)C,(size_t)H,(size_t)W};
    e = clEnqueueNDRangeKernel(q, k, 3, NULL, g, NULL, 0, NULL, NULL);
    check(e,"enqueue add");
}

// Pool2D：mode=0(max),1(avg)；count_include_pad: 0/1
static void run_pool2d(cl_command_queue q, cl_kernel k, cl_mem x, cl_mem y,
                       int C,int IH,int IW,int OH,int OW,int KH,int KW,int SH,int SW,int PH,int PW,int mode,int count_include_pad)
{
    int arg=0; cl_int e=CL_SUCCESS;
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&x);
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&y);
    e|=clSetKernelArg(k,arg++,sizeof(int),&C);
    e|=clSetKernelArg(k,arg++,sizeof(int),&IH);
    e|=clSetKernelArg(k,arg++,sizeof(int),&IW);
    e|=clSetKernelArg(k,arg++,sizeof(int),&OH);
    e|=clSetKernelArg(k,arg++,sizeof(int),&OW);
    e|=clSetKernelArg(k,arg++,sizeof(int),&KH);
    e|=clSetKernelArg(k,arg++,sizeof(int),&KW);
    e|=clSetKernelArg(k,arg++,sizeof(int),&SH);
    e|=clSetKernelArg(k,arg++,sizeof(int),&SW);
    e|=clSetKernelArg(k,arg++,sizeof(int),&PH);
    e|=clSetKernelArg(k,arg++,sizeof(int),&PW);
    e|=clSetKernelArg(k,arg++,sizeof(int),&mode);
    e|=clSetKernelArg(k,arg++,sizeof(int),&count_include_pad);
    check(e,"set pool args");
    size_t g[3]={(size_t)C,(size_t)OH,(size_t)OW};
    e = clEnqueueNDRangeKernel(q, k, 3, NULL, g, NULL, 0, NULL, NULL);
    check(e,"enqueue pool");
}

// GEMM：C[MxN] = A[MxK] * B[KxN]
static void run_gemm(cl_command_queue q, cl_kernel k,
                     cl_mem A, cl_mem B, cl_mem C,
                     int M,int K,int N, int transA, int transB)
{
    int arg=0; cl_int e=CL_SUCCESS;
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&A);
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&B);
    e|=clSetKernelArg(k,arg++,sizeof(cl_mem),&C);
    e|=clSetKernelArg(k,arg++,sizeof(int),&M);
    e|=clSetKernelArg(k,arg++,sizeof(int),&K);
    e|=clSetKernelArg(k,arg++,sizeof(int),&N);
    e|=clSetKernelArg(k,arg++,sizeof(int),&transA);
    e|=clSetKernelArg(k,arg++,sizeof(int),&transB);
    check(e,"set gemm args");
    size_t g[2]={(size_t)M,(size_t)N};
    e = clEnqueueNDRangeKernel(q, k, 2, NULL, g, NULL, 0, NULL, NULL);
    check(e,"enqueue gemm");
}

// ========== ResNet18 结构 ==========
struct ConvBNParams {
    int IC=0,OC=0,KH=0,KW=0,SH=1,SW=1,PH=0,PW=0;
    std::vector<float> W, B, gamma, beta, mean, var;
    cl_mem dW=nullptr,dB=nullptr,dG=nullptr,dBt=nullptr,dMean=nullptr,dVar=nullptr;
};

// 计算输出尺寸
static inline int OUT_H(int IH,int KH,int SH,int PH){ return (IH + 2*PH - KH)/SH + 1; }
static inline int OUT_W(int IW,int KW,int SW,int PW){ return (IW + 2*PW - KW)/SW + 1; }

// 固定种子初始化 Conv+BN
static void init_convbn_seeded(ConvBNParams& p, uint32_t seed_base, uint32_t tag){
    p.W.resize((size_t)p.OC * p.IC * p.KH * p.KW);
    p.B.resize(p.OC);
    p.gamma.resize(p.OC);
    p.beta.resize(p.OC);
    p.mean.resize(p.OC);
    p.var.resize(p.OC);

    fill_uniform(p.W,   -0.1f, 0.1f, seed_mix(seed_base, tag+1));
    fill_uniform(p.B,   -0.1f, 0.1f, seed_mix(seed_base, tag+2));
    std::vector<float> tmp(p.OC);
    fill_uniform(tmp,   -0.1f, 0.1f, seed_mix(seed_base, tag+3));
    for(int i=0;i<p.OC;++i) p.gamma[i] = 1.0f + tmp[i];
    fill_uniform(p.beta, -0.1f, 0.1f,  seed_mix(seed_base, tag+4));
    fill_uniform(p.mean, -0.1f, 0.1f,  seed_mix(seed_base, tag+5));
    fill_uniform(p.var,   0.9f, 1.1f,  seed_mix(seed_base, tag+6));
}

// 上传到设备
static void upload_convbn(cl_context ctx, const ConvBNParams& p, ConvBNParams& d, cl_int* err=nullptr){
    d = p; // 结构复制
    d.dW    = make_device_buffer(ctx, p.W.data(),     sizeof(float)*p.W.size(),     err);
    d.dB    = make_device_buffer(ctx, p.B.data(),     sizeof(float)*p.B.size(),     err);
    d.dG    = make_device_buffer(ctx, p.gamma.data(), sizeof(float)*p.gamma.size(), err);
    d.dBt   = make_device_buffer(ctx, p.beta.data(),  sizeof(float)*p.beta.size(),  err);
    d.dMean = make_device_buffer(ctx, p.mean.data(),  sizeof(float)*p.mean.size(),  err);
    d.dVar  = make_device_buffer(ctx, p.var.data(),   sizeof(float)*p.var.size(),   err);
}

static Tensor make_device_tensor_seeded(cl_context ctx, uint32_t seed, int C,int H,int W, float lo=-1.f, float hi=1.f){
    Tensor t(C,H,W);
    fill_uniform(t.host, lo, hi, seed);
    t.buf = make_device_buffer(ctx, t.host.data(), t.bytes(), nullptr);
    return t;
}
static Tensor empty_device_tensor(cl_context ctx, int C,int H,int W){
    Tensor t(C,H,W);
    t.buf = make_device_buffer(ctx, nullptr, t.bytes(), nullptr);
    return t;
}

// BasicBlock（包含可选下采样1x1）
struct BasicBlock {
    ConvBNParams conv1, conv2;
    bool has_down=false;
    ConvBNParams down;

    Tensor forward(cl_command_queue q, Kernels& K, cl_context ctx, const Tensor& x){
        // conv1
        int oh1 = OUT_H(x.H, conv1.KH, conv1.SH, conv1.PH);
        int ow1 = OUT_W(x.W, conv1.KW, conv1.SW, conv1.PW);
        Tensor y1 = empty_device_tensor(ctx, conv1.OC, oh1, ow1);
        zero_buffer(q, y1.buf, y1.bytes());
        run_conv_bn(q, K.k_convbn, x.buf,
                    conv1.dW, conv1.dB, conv1.dG, conv1.dBt, conv1.dMean, conv1.dVar, BN_EPS, y1.buf,
                    conv1.IC, x.H, x.W, conv1.OC, conv1.KH, conv1.KW, oh1, ow1, conv1.SH, conv1.SW, conv1.PH, conv1.PW);

        Tensor y1r = empty_device_tensor(ctx, y1.C, y1.H, y1.W);
        zero_buffer(q, y1r.buf, y1r.bytes());
        run_relu(q, K.k_relu, y1.buf, y1r.buf, y1.C, y1.H, y1.W);

        // conv2
        int oh2 = OUT_H(y1r.H, conv2.KH, conv2.SH, conv2.PH);
        int ow2 = OUT_W(y1r.W, conv2.KW, conv2.SW, conv2.PW);
        Tensor y2 = empty_device_tensor(ctx, conv2.OC, oh2, ow2);
        zero_buffer(q, y2.buf, y2.bytes());
        run_conv_bn(q, K.k_convbn, y1r.buf,
                    conv2.dW, conv2.dB, conv2.dG, conv2.dBt, conv2.dMean, conv2.dVar, BN_EPS, y2.buf,
                    conv2.IC, y1r.H, y1r.W, conv2.OC, conv2.KH, conv2.KW, oh2, ow2, conv2.SH, conv2.SW, conv2.PH, conv2.PW);

        // identity（下采样？）
        Tensor id;
        bool need_release_id = false;
        if (has_down){
            int odh = OUT_H(x.H, down.KH, down.SH, down.PH);
            int odw = OUT_W(x.W, down.KW, down.SW, down.PW);
            id = empty_device_tensor(ctx, down.OC, odh, odw);
            zero_buffer(q, id.buf, id.bytes());
            run_conv_bn(q, K.k_convbn, x.buf,
                        down.dW, down.dB, down.dG, down.dBt, down.dMean, down.dVar, BN_EPS, id.buf,
                        down.IC, x.H, x.W, down.OC, down.KH, down.KW, odh, odw, down.SH, down.SW, down.PH, down.PW);
            need_release_id = true;
        } else {
            id = x; // alias，不释放
        }

        // add + relu
        Tensor y = empty_device_tensor(ctx, y2.C, y2.H, y2.W);
        zero_buffer(q, y.buf, y.bytes());
        run_add(q, K.k_add, y2.buf, id.buf, y.buf, y.C, y.H, y.W);

        Tensor yr = empty_device_tensor(ctx, y.C, y.H, y.W);
        zero_buffer(q, yr.buf, yr.bytes());
        run_relu(q, K.k_relu, y.buf, yr.buf, y.C, y.H, y.W);

        // 确保上述 kernel 完成，再释放临时
        clFinish(q);
        safe_release_mem(y1.buf);
        safe_release_mem(y1r.buf);
        safe_release_mem(y2.buf);
        safe_release_mem(y.buf);
        if (need_release_id) safe_release_mem(id.buf);

        return yr; // 返回的 yr.buf 由调用者负责释放
    }
};

static BasicBlock make_basic(int inC, int outC, int stride){
    BasicBlock b;
    b.conv1.IC=inC;  b.conv1.OC=outC; b.conv1.KH=3; b.conv1.KW=3;
    b.conv1.SH=stride; b.conv1.SW=stride; b.conv1.PH=1; b.conv1.PW=1;
    b.conv2.IC=outC; b.conv2.OC=outC; b.conv2.KH=3; b.conv2.KW=3;
    b.conv2.SH=1; b.conv2.SW=1; b.conv2.PH=1; b.conv2.PW=1;
    b.has_down = (stride!=1) || (inC!=outC);
    if (b.has_down){
        b.down.IC=inC; b.down.OC=outC; b.down.KH=1; b.down.KW=1;
        b.down.SH=stride; b.down.SW=stride; b.down.PH=0; b.down.PW=0;
    }
    return b;
}

int main(){
    const uint32_t SEED_BASE = 0xCAFED00Du;

    // OpenCL 初始化
    cl_int err=CL_SUCCESS;
    cl_platform_id plat; cl_device_id dev;
    err = clGetPlatformIDs(1,&plat,NULL);              check(err,"clGetPlatformIDs");
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_DEFAULT, 1,&dev,NULL); check(err,"clGetDeviceIDs");
    cl_context ctx = clCreateContext(NULL,1,&dev,NULL,NULL,&err);     check(err,"clCreateContext");
    cl_command_queue q = clCreateCommandQueue(ctx,dev,0,&err);        check(err,"clCreateCommandQueue");

    Kernels K; build_all_kernels(ctx, dev, K);

    // ===== 构建 ResNet18（固定种子初始化 + 上传） =====
    // stem: conv7x7 s2 p3
    ConvBNParams stem; stem.IC=IN_C; stem.OC=64; stem.KH=7; stem.KW=7; stem.SH=2; stem.SW=2; stem.PH=3; stem.PW=3;
    init_convbn_seeded(stem, SEED_BASE, 100);
    ConvBNParams d_stem; upload_convbn(ctx, stem, d_stem, &err);

    auto init_upload_block = [&](BasicBlock& b, BasicBlock& db, uint32_t tag){
        init_convbn_seeded(b.conv1, SEED_BASE, tag+1);
        init_convbn_seeded(b.conv2, SEED_BASE, tag+2);
        if (b.has_down) init_convbn_seeded(b.down, SEED_BASE, tag+3);
        upload_convbn(ctx, b.conv1, db.conv1, &err);
        upload_convbn(ctx, b.conv2, db.conv2, &err);
        db.has_down = b.has_down;
        if (b.has_down) upload_convbn(ctx, b.down, db.down, &err);
    };

    // stages
    BasicBlock st2_b1 = make_basic(64,  64, 1), st2_b2 = make_basic(64,  64, 1);
    BasicBlock st3_b1 = make_basic(64, 128, 2), st3_b2 = make_basic(128, 128,1);
    BasicBlock st4_b1 = make_basic(128,256, 2), st4_b2 = make_basic(256, 256,1);
    BasicBlock st5_b1 = make_basic(256,512, 2), st5_b2 = make_basic(512, 512,1);

    BasicBlock d_st2_b1, d_st2_b2, d_st3_b1, d_st3_b2, d_st4_b1, d_st4_b2, d_st5_b1, d_st5_b2;
    init_upload_block(st2_b1, d_st2_b1, 2000);
    init_upload_block(st2_b2, d_st2_b2, 2100);
    init_upload_block(st3_b1, d_st3_b1, 2200);
    init_upload_block(st3_b2, d_st3_b2, 2300);
    init_upload_block(st4_b1, d_st4_b1, 2400);
    init_upload_block(st4_b2, d_st4_b2, 2500);
    init_upload_block(st5_b1, d_st5_b1, 2600);
    init_upload_block(st5_b2, d_st5_b2, 2700);

    // FC
    std::vector<float> fcW((size_t)512*NUM_CLASSES), fcB(NUM_CLASSES);
    fill_uniform(fcW, -0.1f, 0.1f, seed_mix(SEED_BASE, 3001));
    fill_uniform(fcB, -0.1f, 0.1f, seed_mix(SEED_BASE, 3002));
    cl_mem d_fcW = make_device_buffer(ctx, fcW.data(), sizeof(float)*fcW.size(), &err);
    cl_mem d_fcB = make_device_buffer(ctx, fcB.data(), sizeof(float)*fcB.size(), &err);

    // 输入（固定种子）
    Tensor x0; x0 = make_device_tensor_seeded(ctx, seed_mix(SEED_BASE, 1), IN_C, IN_H, IN_W, -1.f, 1.f);

    // ===== Forward =====
    // stem forward
    int oh = OUT_H(IN_H, stem.KH, stem.SH, stem.PH);
    int ow = OUT_W(IN_W, stem.KW, stem.SW, stem.PW);
    Tensor s1 = empty_device_tensor(ctx, stem.OC, oh, ow);
    zero_buffer(q, s1.buf, s1.bytes());
    run_conv_bn(q, K.k_convbn, x0.buf,
                d_stem.dW, d_stem.dB, d_stem.dG, d_stem.dBt, d_stem.dMean, d_stem.dVar, BN_EPS, s1.buf,
                stem.IC, IN_H, IN_W, stem.OC, stem.KH, stem.KW, oh, ow, stem.SH, stem.SW, stem.PH, stem.PW);
    Tensor s1r = empty_device_tensor(ctx, s1.C, s1.H, s1.W);
    zero_buffer(q, s1r.buf, s1r.bytes());
    run_relu(q, K.k_relu, s1.buf, s1r.buf, s1r.C, s1r.H, s1r.W);

    // maxpool 3x3 s2 p1
    int mph=3, mpw=3, mps=2, mpp=1;
    int poh = OUT_H(s1r.H, mph, mps, mpp);
    int pow = OUT_W(s1r.W, mpw, mps, mpp);
    Tensor p1 = empty_device_tensor(ctx, s1r.C, poh, pow);
    zero_buffer(q, p1.buf, p1.bytes());
    run_pool2d(q, K.k_pool, s1r.buf, p1.buf, p1.C, s1r.H, s1r.W, poh, pow, mph, mpw, mps, mps, mpp, mpp, 0, 0);

    // 释放 stem 中间
    clFinish(q);
    safe_release_mem(s1.buf);
    safe_release_mem(s1r.buf);

    // stages 链式
    Tensor t = p1, tmp;
    tmp = d_st2_b1.forward(q, K, ctx, t); safe_release_mem(t.buf); t = tmp; p1.buf=nullptr; // 防止尾部再次释放 p1
    tmp = d_st2_b2.forward(q, K, ctx, t); safe_release_mem(t.buf); t = tmp;
    tmp = d_st3_b1.forward(q, K, ctx, t); safe_release_mem(t.buf); t = tmp;
    tmp = d_st3_b2.forward(q, K, ctx, t); safe_release_mem(t.buf); t = tmp;
    tmp = d_st4_b1.forward(q, K, ctx, t); safe_release_mem(t.buf); t = tmp;
    tmp = d_st4_b2.forward(q, K, ctx, t); safe_release_mem(t.buf); t = tmp;
    tmp = d_st5_b1.forward(q, K, ctx, t); safe_release_mem(t.buf); t = tmp;
    tmp = d_st5_b2.forward(q, K, ctx, t); safe_release_mem(t.buf); t = tmp;

    // global avg pool -> 1x1
    Tensor g = empty_device_tensor(ctx, t.C, 1, 1);
    zero_buffer(q, g.buf, g.bytes());
    run_pool2d(q, K.k_pool, t.buf, g.buf, g.C, t.H, t.W, 1, 1, t.H, t.W, 1, 1, 0, 0, 1, 0);

    // GEMM: A(1x512)=g, B(512xNUM_CLASSES)=fcW -> C(1xNUM_CLASSES)
    cl_mem d_logits = make_device_buffer(ctx, nullptr, sizeof(float)*NUM_CLASSES, &err);
    zero_buffer(q, d_logits, sizeof(float)*NUM_CLASSES);
    int M=1, Kdim=g.C, Ncls=NUM_CLASSES, transA=0, transB=0;
    run_gemm(q, K.k_gemm, g.buf, d_fcW, d_logits, M, Kdim, Ncls, transA, transB);

    clFinish(q);

    // 读回 logits 并加偏置
    std::vector<float> logits(NUM_CLASSES, 0.0f);
    download_tensor(q, d_logits, logits.data(), sizeof(float)*NUM_CLASSES);
    for (int j=0;j<NUM_CLASSES;++j) logits[j]+=fcB[j];

    puts("\nResNet18 logits (first 10, deterministic):");
    for (int i=0;i<10 && i<NUM_CLASSES;++i)
        printf("logit[%d] = %+9.6f\n", i, logits[i]);

    // ===== 清理（安全释放）=====
    // 中间张量
    safe_release_mem(x0.buf);
    safe_release_mem(p1.buf);   // 已置空，safe
    safe_release_mem(t.buf);
    safe_release_mem(g.buf);
    safe_release_mem(d_fcW);
    safe_release_mem(d_fcB);
    safe_release_mem(d_logits);

    // 释放卷积权重
    auto rel_convbn = [&](ConvBNParams& p){
        safe_release_mem(p.dW);
        safe_release_mem(p.dB);
        safe_release_mem(p.dG);
        safe_release_mem(p.dBt);
        safe_release_mem(p.dMean);
        safe_release_mem(p.dVar);
    };
    rel_convbn(d_stem);
    rel_convbn(d_st2_b1.conv1); rel_convbn(d_st2_b1.conv2); if(d_st2_b1.has_down) rel_convbn(d_st2_b1.down);
    rel_convbn(d_st2_b2.conv1); rel_convbn(d_st2_b2.conv2); if(d_st2_b2.has_down) rel_convbn(d_st2_b2.down);
    rel_convbn(d_st3_b1.conv1); rel_convbn(d_st3_b1.conv2); if(d_st3_b1.has_down) rel_convbn(d_st3_b1.down);
    rel_convbn(d_st3_b2.conv1); rel_convbn(d_st3_b2.conv2); if(d_st3_b2.has_down) rel_convbn(d_st3_b2.down);
    rel_convbn(d_st4_b1.conv1); rel_convbn(d_st4_b1.conv2); if(d_st4_b1.has_down) rel_convbn(d_st4_b1.down);
    rel_convbn(d_st4_b2.conv1); rel_convbn(d_st4_b2.conv2); if(d_st4_b2.has_down) rel_convbn(d_st4_b2.down);
    rel_convbn(d_st5_b1.conv1); rel_convbn(d_st5_b1.conv2); if(d_st5_b1.has_down) rel_convbn(d_st5_b1.down);
    rel_convbn(d_st5_b2.conv1); rel_convbn(d_st5_b2.conv2); if(d_st5_b2.has_down) rel_convbn(d_st5_b2.down);

    // kernels & programs
    if (K.k_conv)    clReleaseKernel(K.k_conv);
    if (K.k_convbn)  clReleaseKernel(K.k_convbn);
    if (K.k_bn)      clReleaseKernel(K.k_bn);
    clReleaseKernel(K.k_relu);
    clReleaseKernel(K.k_add);
    clReleaseKernel(K.k_pool);
    clReleaseKernel(K.k_gemm);

    if (K.prog_conv)   clReleaseProgram(K.prog_conv);
    if (K.prog_convbn) clReleaseProgram(K.prog_convbn);
    if (K.prog_bn)     clReleaseProgram(K.prog_bn);
    clReleaseProgram(K.prog_relu);
    clReleaseProgram(K.prog_add);
    clReleaseProgram(K.prog_pool);
    clReleaseProgram(K.prog_gemm);

    // queue & context
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return 0;
}
