#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <cassert>
#include <iostream>

// ======= 路径（相对于本文件所在工作目录）=======
static const char* KPATH_CONV      = "../AIops/Conv/conv.cl";
static const char* KPATH_CONV_BN   = "../AIops/Conv_BN/conv_bn.cl";
static const char* KPATH_BN        = "../AIops/BatchNorm2d/bn2d.cl";
static const char* KPATH_RELU      = "../AIops/ReLU/relu.cl";
static const char* KPATH_ADD       = "../AIops/add/add.cl";
static const char* KPATH_POOL2D    = "../AIops/Pool2D/pool2d.cl";
static const char* KPATH_GEMM      = "../AIops/GEMM/gemm.cl";

// ======= 配置 =======
static const int  NUM_CLASSES = 10;
static const int  N = 1;                // Batch 固定 1，布局 NCHW
static const int  IN_C = 3, IN_H = 224/4, IN_W = 224/4;
static const float BN_EPS = 1e-5f;
// 这里采用不融合 BN 的版本，AlexNet 原版也无 BN
static const bool  USE_FUSED_CONV_BN = false;

// ======= 简单工具 =======
static inline float frand() { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }

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

struct Tensor {
    int C,H,W;             // N=1 固定
    std::vector<float> host;
    cl_mem buf = nullptr;
    Tensor(){}
    Tensor(int c,int h,int w):C(c),H(h),W(w),host((size_t)c*h*w) {}
    size_t bytes() const { return (size_t)C*H*W*sizeof(float); }
    size_t elems() const { return (size_t)C*H*W; }
};

// 创建并上传
static cl_mem make_device_buffer(cl_context ctx, const void* host, size_t bytes, cl_int* err=nullptr){
    cl_int e;
    cl_mem b = clCreateBuffer(ctx, CL_MEM_READ_WRITE | (host?CL_MEM_COPY_HOST_PTR:0), bytes, (void*)host, &e);
    if(err) *err=e;
    if(e!=CL_SUCCESS){ fprintf(stderr,"clCreateBuffer failed: %d\n", e); exit(1); }
    return b;
}

static void upload_tensor(cl_command_queue q, cl_mem buf, const void* src, size_t bytes){
    cl_int e = clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, bytes, src, 0, NULL, NULL);
    if (e!=CL_SUCCESS){ fprintf(stderr,"EnqueueWriteBuffer failed: %d\n", e); exit(1); }
}
static void download_tensor(cl_command_queue q, cl_mem buf, void* dst, size_t bytes){
    cl_int e = clEnqueueReadBuffer(q, buf, CL_TRUE, 0, bytes, dst, 0, NULL, NULL);
    if (e!=CL_SUCCESS){ fprintf(stderr,"EnqueueReadBuffer failed: %d\n", e); exit(1); }
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
        K.k_convbn    = clCreateKernel(K.prog_convbn, "conv2d_bn", &err);
        if (err!=CL_SUCCESS){ fprintf(stderr,"clCreateKernel(conv2d_bn) err=%d\n", err); exit(1); }
    } else {
        K.prog_conv   = build_program(ctx, dev, KPATH_CONV);
        K.k_conv      = clCreateKernel(K.prog_conv, "conv2d", &err);   // <<<<<< 这里
        if (err!=CL_SUCCESS){ fprintf(stderr,"clCreateKernel(conv2d) err=%d\n", err); exit(1); }
        // 如需 BN 再启用
        // K.prog_bn   = build_program(ctx, dev, KPATH_BN);
        // K.k_bn      = clCreateKernel(K.prog_bn, "batchnorm2d_infer", &err);
        // if (err!=CL_SUCCESS){ ... }
    }

    K.prog_relu   = build_program(ctx, dev, KPATH_RELU);
    K.k_relu      = clCreateKernel(K.prog_relu, "relu3d", &err);
    if (err!=CL_SUCCESS){ fprintf(stderr,"clCreateKernel(relu3d) err=%d\n", err); exit(1); }

    K.prog_add    = build_program(ctx, dev, KPATH_ADD);
    K.k_add       = clCreateKernel(K.prog_add, "add3d", &err);
    if (err!=CL_SUCCESS){ fprintf(stderr,"clCreateKernel(add3d) err=%d\n", err); exit(1); }

    K.prog_pool   = build_program(ctx, dev, KPATH_POOL2D);
    K.k_pool      = clCreateKernel(K.prog_pool, "pool2d", &err);
    if (err!=CL_SUCCESS){ fprintf(stderr,"clCreateKernel(pool2d) err=%d\n", err); exit(1); }

    K.prog_gemm   = build_program(ctx, dev, KPATH_GEMM);
    K.k_gemm      = clCreateKernel(K.prog_gemm, "gemm", &err);
    if (err!=CL_SUCCESS){ fprintf(stderr,"clCreateKernel(gemm) err=%d\n", err); exit(1); }
}


// ======= AlexNet 卷积参数 =======
struct ConvParams {
    int IC, OC, KH, KW, SH, SW, PH, PW;
    std::vector<float> W, B;
    cl_mem dW=nullptr, dB=nullptr;
};

static void init_conv(ConvParams& p) {
    p.W.resize((size_t)p.OC * p.IC * p.KH * p.KW);
    p.B.resize(p.OC);
    for (auto &v : p.W) v = frand() * 0.1f;
    for (int i = 0; i < p.OC; ++i) p.B[i] = frand() * 0.1f;
}

static inline int OUT_H(int IH,int KH,int SH,int PH) { return (IH + 2*PH - KH) / SH + 1; }
static inline int OUT_W(int IW,int KW,int SW,int PW) { return (IW + 2*PW - KW) / SW + 1; }

// ======= 一些运行封装 =======
static void run_conv_layer(cl_command_queue q, cl_kernel k_conv,
                           cl_mem in, cl_mem w, cl_mem b, cl_mem out,
                           int IC, int IH, int IW,
                           int OC, int KH, int KW,
                           int OH, int OW, int SH, int SW, int PH, int PW,
                           int do_relu) // 新增
{
    int arg = 0;
    clSetKernelArg(k_conv, arg++, sizeof(cl_mem), &in);
    clSetKernelArg(k_conv, arg++, sizeof(cl_mem), &w);
    clSetKernelArg(k_conv, arg++, sizeof(cl_mem), &b);
    clSetKernelArg(k_conv, arg++, sizeof(cl_mem), &out);
    clSetKernelArg(k_conv, arg++, sizeof(int), &IC);
    clSetKernelArg(k_conv, arg++, sizeof(int), &IH);
    clSetKernelArg(k_conv, arg++, sizeof(int), &IW);
    clSetKernelArg(k_conv, arg++, sizeof(int), &OC);
    clSetKernelArg(k_conv, arg++, sizeof(int), &KH);
    clSetKernelArg(k_conv, arg++, sizeof(int), &KW);
    clSetKernelArg(k_conv, arg++, sizeof(int), &OH);
    clSetKernelArg(k_conv, arg++, sizeof(int), &OW);
    clSetKernelArg(k_conv, arg++, sizeof(int), &SH);
    clSetKernelArg(k_conv, arg++, sizeof(int), &SW);
    clSetKernelArg(k_conv, arg++, sizeof(int), &PH);
    clSetKernelArg(k_conv, arg++, sizeof(int), &PW);
    clSetKernelArg(k_conv, arg++, sizeof(int), &do_relu); // 传 0 或 1

    size_t g[3] = {(size_t)OC, (size_t)OH, (size_t)OW};  // <<<<<< 顺序改正
    cl_int e = clEnqueueNDRangeKernel(q, k_conv, 3, NULL, g, NULL, 0, NULL, NULL);
    if (e!=CL_SUCCESS){ fprintf(stderr,"Enqueue conv failed: %d\n", e); exit(1); }
}


static void run_relu3d(cl_command_queue q, cl_kernel k_relu,
                       cl_mem in, cl_mem out, int C, int H, int W)
{
    int arg=0;
    clSetKernelArg(k_relu, arg++, sizeof(cl_mem), &in);
    clSetKernelArg(k_relu, arg++, sizeof(cl_mem), &out);
    clSetKernelArg(k_relu, arg++, sizeof(int), &C);
    clSetKernelArg(k_relu, arg++, sizeof(int), &H);
    clSetKernelArg(k_relu, arg++, sizeof(int), &W);
    size_t g[3] = {(size_t)C, (size_t)H, (size_t)W};   // <<<<<<
    cl_int e = clEnqueueNDRangeKernel(q, k_relu, 3, NULL, g, NULL, 0, NULL, NULL);
    if (e!=CL_SUCCESS){ fprintf(stderr,"Enqueue relu failed: %d\n", e); exit(1); }
}

static void run_pool2d(cl_command_queue q, cl_kernel k_pool,
                       cl_mem in, cl_mem out, int C, int IH, int IW,
                       int OH, int OW, int KH, int KW, int SH, int SW, int PH, int PW,
                       int mode, int count_include_pad)
{
    int arg=0;
    clSetKernelArg(k_pool, arg++, sizeof(cl_mem), &in);
    clSetKernelArg(k_pool, arg++, sizeof(cl_mem), &out);
    clSetKernelArg(k_pool, arg++, sizeof(int), &C);
    clSetKernelArg(k_pool, arg++, sizeof(int), &IH);
    clSetKernelArg(k_pool, arg++, sizeof(int), &IW);
    clSetKernelArg(k_pool, arg++, sizeof(int), &OH);
    clSetKernelArg(k_pool, arg++, sizeof(int), &OW);
    clSetKernelArg(k_pool, arg++, sizeof(int), &KH);
    clSetKernelArg(k_pool, arg++, sizeof(int), &KW);
    clSetKernelArg(k_pool, arg++, sizeof(int), &SH);
    clSetKernelArg(k_pool, arg++, sizeof(int), &SW);
    clSetKernelArg(k_pool, arg++, sizeof(int), &PH);
    clSetKernelArg(k_pool, arg++, sizeof(int), &PW);
    clSetKernelArg(k_pool, arg++, sizeof(int), &mode);
    clSetKernelArg(k_pool, arg++, sizeof(int), &count_include_pad);

    size_t g[3] = {(size_t)C, (size_t)OH, (size_t)OW}; // <<<<<<
    cl_int e = clEnqueueNDRangeKernel(q, k_pool, 3, NULL, g, NULL, 0, NULL, NULL);
    if (e!=CL_SUCCESS){ fprintf(stderr,"Enqueue pool failed: %d\n", e); exit(1); }
}


static void run_gemm(cl_command_queue q, cl_kernel k_gemm,
                     cl_mem A, cl_mem B, cl_mem C,
                     int M, int K, int N,
                     int do_trans_a, int do_trans_b)
{
    int arg=0;
    clSetKernelArg(k_gemm, arg++, sizeof(cl_mem), &A);
    clSetKernelArg(k_gemm, arg++, sizeof(cl_mem), &B);
    clSetKernelArg(k_gemm, arg++, sizeof(cl_mem), &C);
    clSetKernelArg(k_gemm, arg++, sizeof(int), &M);
    clSetKernelArg(k_gemm, arg++, sizeof(int), &K);
    clSetKernelArg(k_gemm, arg++, sizeof(int), &N);
    clSetKernelArg(k_gemm, arg++, sizeof(int), &do_trans_a);
    clSetKernelArg(k_gemm, arg++, sizeof(int), &do_trans_b);
    size_t g[2] = {(size_t)M, (size_t)N}; // row=M 在 dim0, col=N 在 dim1
    cl_int e = clEnqueueNDRangeKernel(q, k_gemm, 2, NULL, g, NULL, 0, NULL, NULL);
    if (e!=CL_SUCCESS){ fprintf(stderr,"Enqueue gemm failed: %d\n", e); exit(1); }
}

static void run_add3d(cl_command_queue q, cl_kernel k_add,
                      cl_mem a, cl_mem b, cl_mem out, int C, int H, int W)
{
    int arg=0;
    clSetKernelArg(k_add, arg++, sizeof(cl_mem), &a);
    clSetKernelArg(k_add, arg++, sizeof(cl_mem), &b);
    clSetKernelArg(k_add, arg++, sizeof(cl_mem), &out);
    clSetKernelArg(k_add, arg++, sizeof(int), &C);
    clSetKernelArg(k_add, arg++, sizeof(int), &H);
    clSetKernelArg(k_add, arg++, sizeof(int), &W);
    size_t g[3] = {(size_t)C, (size_t)H, (size_t)W};
    cl_int e = clEnqueueNDRangeKernel(q, k_add, 3, NULL, g, NULL, 0, NULL, NULL);
    if (e!=CL_SUCCESS){ fprintf(stderr,"Enqueue add3d failed: %d\n", e); exit(1); }
}


int main() {
    srand((unsigned)time(NULL));

    // OpenCL setup
    cl_int err = CL_SUCCESS;
    cl_platform_id plat; cl_device_id dev;
    err = clGetPlatformIDs(1, &plat, NULL);
    if (err!=CL_SUCCESS){ fprintf(stderr,"clGetPlatformIDs err=%d\n", err); return -1; }
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err!=CL_SUCCESS){ // 兜底用 CPU
        fprintf(stderr,"Use CPU device, err=%d\n", err);
        err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
        if (err!=CL_SUCCESS){ fprintf(stderr,"clGetDeviceIDs err=%d\n", err); return -1; }
    }

    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)plat, 0};
    cl_context ctx = clCreateContext(properties, 1, &dev, NULL, NULL, &err);
    if (err!=CL_SUCCESS){ fprintf(stderr,"clCreateContext err=%d\n", err); return -1; }

    cl_command_queue_properties queue_properties = 0;
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, &queue_properties, &err);
    if (err!=CL_SUCCESS){ fprintf(stderr,"clCreateCommandQueueWithProperties err=%d\n", err); return -1; }

    Kernels K; build_all_kernels(ctx, dev, K);

    // ===== 构建 AlexNet =====
    // conv1: 11x11, s=4, p=2
    ConvParams conv1; conv1.IC = IN_C; conv1.OC = 64; conv1.KH = 11; conv1.KW = 11; conv1.SH = 4; conv1.SW = 4; conv1.PH = 2; conv1.PW = 2;
    init_conv(conv1);

    // conv2: 5x5, s=1, p=2
    ConvParams conv2; conv2.IC = 64; conv2.OC = 192; conv2.KH = 5; conv2.KW = 5; conv2.SH = 1; conv2.SW = 1; conv2.PH = 2; conv2.PW = 2;
    init_conv(conv2);

    // conv3: 3x3, s=1, p=1
    ConvParams conv3; conv3.IC = 192; conv3.OC = 384; conv3.KH = 3; conv3.KW = 3; conv3.SH = 1; conv3.SW = 1; conv3.PH = 1; conv3.PW = 1;
    init_conv(conv3);

    // conv4: 3x3, s=1, p=1
    ConvParams conv4; conv4.IC = 384; conv4.OC = 256; conv4.KH = 3; conv4.KW = 3; conv4.SH = 1; conv4.SW = 1; conv4.PH = 1; conv4.PW = 1;
    init_conv(conv4);

    // conv5: 3x3, s=1, p=1
    ConvParams conv5; conv5.IC = 256; conv5.OC = 256; conv5.KH = 3; conv5.KW = 3; conv5.SH = 1; conv5.SW = 1; conv5.PH = 1; conv5.PW = 1;
    init_conv(conv5);

    // 上传卷积权重
    conv1.dW = make_device_buffer(ctx, conv1.W.data(), sizeof(float)*conv1.W.size(), &err);
    conv1.dB = make_device_buffer(ctx, conv1.B.data(), sizeof(float)*conv1.B.size(), &err);
    conv2.dW = make_device_buffer(ctx, conv2.W.data(), sizeof(float)*conv2.W.size(), &err);
    conv2.dB = make_device_buffer(ctx, conv2.B.data(), sizeof(float)*conv2.B.size(), &err);
    conv3.dW = make_device_buffer(ctx, conv3.W.data(), sizeof(float)*conv3.W.size(), &err);
    conv3.dB = make_device_buffer(ctx, conv3.B.data(), sizeof(float)*conv3.B.size(), &err);
    conv4.dW = make_device_buffer(ctx, conv4.W.data(), sizeof(float)*conv4.W.size(), &err);
    conv4.dB = make_device_buffer(ctx, conv4.B.data(), sizeof(float)*conv4.B.size(), &err);
    conv5.dW = make_device_buffer(ctx, conv5.W.data(), sizeof(float)*conv5.W.size(), &err);
    conv5.dB = make_device_buffer(ctx, conv5.B.data(), sizeof(float)*conv5.B.size(), &err);

    // 全连接层
    const int FC1_SIZE = 4096;
    const int FC2_SIZE = 4096;
    std::vector<float> fcW1(FC1_SIZE * 256 * 6 * 6), fcB1(FC1_SIZE);
    std::vector<float> fcW2(FC2_SIZE * FC1_SIZE), fcB2(FC2_SIZE);
    std::vector<float> fcW3(NUM_CLASSES * FC2_SIZE), fcB3(NUM_CLASSES);
    for (auto &v : fcW1) v = frand() * 0.1f;
    for (auto &v : fcB1) v = frand() * 0.1f;
    for (auto &v : fcW2) v = frand() * 0.1f;
    for (auto &v : fcB2) v = frand() * 0.1f;
    for (auto &v : fcW3) v = frand() * 0.1f;
    for (auto &v : fcB3) v = frand() * 0.1f;

    cl_mem d_fcW1 = make_device_buffer(ctx, fcW1.data(), sizeof(float) * fcW1.size(), &err);
    cl_mem d_fcB1 = make_device_buffer(ctx, fcB1.data(), sizeof(float) * fcB1.size(), &err);
    cl_mem d_fcW2 = make_device_buffer(ctx, fcW2.data(), sizeof(float) * fcW2.size(), &err);
    cl_mem d_fcB2 = make_device_buffer(ctx, fcB2.data(), sizeof(float) * fcB2.size(), &err);
    cl_mem d_fcW3 = make_device_buffer(ctx, fcW3.data(), sizeof(float) * fcW3.size(), &err);
    cl_mem d_fcB3 = make_device_buffer(ctx, fcB3.data(), sizeof(float) * fcB3.size(), &err);

    // 输入
    Tensor x0 = Tensor(IN_C, IN_H, IN_W);
    for (auto &v : x0.host) v = frand(); // 随机输入
    x0.buf = make_device_buffer(ctx, x0.host.data(), x0.bytes(), &err);

    // ======== Forward: conv1 -> relu -> pool ========
    Tensor y1( conv1.OC, OUT_H(x0.H, conv1.KH, conv1.SH, conv1.PH), OUT_W(x0.W, conv1.KW, conv1.SW, conv1.PW) );
    y1.buf = make_device_buffer(ctx, nullptr, y1.bytes(), &err);
    run_conv_layer(q, K.k_conv, x0.buf, conv1.dW, conv1.dB, y1.buf,
                   conv1.IC, x0.H, x0.W, conv1.OC, conv1.KH, conv1.KW, y1.H, y1.W,
                   conv1.SH, conv1.SW, conv1.PH, conv1.PW,0);

    Tensor r1 = y1; // ReLU 输出重建一个 Tensor（共享形状）
    r1.buf = make_device_buffer(ctx, nullptr, r1.bytes(), &err);
    run_relu3d(q, K.k_relu, y1.buf, r1.buf, r1.C, r1.H, r1.W);

    // pool1: 3x3 s=2 p=0
    const int P1_K=3, P1_S=2, P_PAD=0;
    Tensor p1( r1.C, OUT_H(r1.H, P1_K, P1_S, P_PAD), OUT_W(r1.W, P1_K, P1_S, P_PAD) );
    p1.buf = make_device_buffer(ctx, nullptr, p1.bytes(), &err);
    // pool1
    run_pool2d(q, K.k_pool, r1.buf, p1.buf, p1.C, r1.H, r1.W, p1.H, p1.W,
           P1_K, P1_K, P1_S, P1_S, P_PAD, P_PAD, 0, 0);

    // ======== conv2 -> relu -> pool ========
    Tensor y2( conv2.OC, OUT_H(p1.H, conv2.KH, conv2.SH, conv2.PH), OUT_W(p1.W, conv2.KW, conv2.SW, conv2.PW) );
    y2.buf = make_device_buffer(ctx, nullptr, y2.bytes(), &err);
    run_conv_layer(q, K.k_conv, p1.buf, conv2.dW, conv2.dB, y2.buf,
                   conv2.IC, p1.H, p1.W, conv2.OC, conv2.KH, conv2.KW, y2.H, y2.W,
                   conv2.SH, conv2.SW, conv2.PH, conv2.PW,0);

    Tensor r2 = y2;
    r2.buf = make_device_buffer(ctx, nullptr, r2.bytes(), &err);
    run_relu3d(q, K.k_relu, y2.buf, r2.buf, r2.C, r2.H, r2.W);

    // pool2: 3x3 s=2 p=0
    Tensor p2( r2.C, OUT_H(r2.H, P1_K, P1_S, P_PAD), OUT_W(r2.W, P1_K, P1_S, P_PAD) );
    p2.buf = make_device_buffer(ctx, nullptr, p2.bytes(), &err);
    run_pool2d(q, K.k_pool, r2.buf, p2.buf, p2.C, r2.H, r2.W, p2.H, p2.W,
           P1_K, P1_K, P1_S, P1_S, P_PAD, P_PAD, 0, 0);

    // ======== conv3 -> relu ========
    Tensor y3( conv3.OC, OUT_H(p2.H, conv3.KH, conv3.SH, conv3.PH), OUT_W(p2.W, conv3.KW, conv3.SW, conv3.PW) );
    y3.buf = make_device_buffer(ctx, nullptr, y3.bytes(), &err);
    run_conv_layer(q, K.k_conv, p2.buf, conv3.dW, conv3.dB, y3.buf,
                   conv3.IC, p2.H, p2.W, conv3.OC, conv3.KH, conv3.KW, y3.H, y3.W,
                   conv3.SH, conv3.SW, conv3.PH, conv3.PW,0);
    Tensor r3 = y3; r3.buf = make_device_buffer(ctx, nullptr, r3.bytes(), &err);
    run_relu3d(q, K.k_relu, y3.buf, r3.buf, r3.C, r3.H, r3.W);

    // ======== conv4 -> relu ========
    Tensor y4( conv4.OC, OUT_H(r3.H, conv4.KH, conv4.SH, conv4.PH), OUT_W(r3.W, conv4.KW, conv4.SW, conv4.PW) );
    y4.buf = make_device_buffer(ctx, nullptr, y4.bytes(), &err);
    run_conv_layer(q, K.k_conv, r3.buf, conv4.dW, conv4.dB, y4.buf,
                   conv4.IC, r3.H, r3.W, conv4.OC, conv4.KH, conv4.KW, y4.H, y4.W,
                   conv4.SH, conv4.SW, conv4.PH, conv4.PW,0);
    Tensor r4 = y4; r4.buf = make_device_buffer(ctx, nullptr, r4.bytes(), &err);
    run_relu3d(q, K.k_relu, y4.buf, r4.buf, r4.C, r4.H, r4.W);

    // ======== conv5 -> relu -> pool ========
    Tensor y5( conv5.OC, OUT_H(r4.H, conv5.KH, conv5.SH, conv5.PH), OUT_W(r4.W, conv5.KW, conv5.SW, conv5.PW) );
    y5.buf = make_device_buffer(ctx, nullptr, y5.bytes(), &err);
    run_conv_layer(q, K.k_conv, r4.buf, conv5.dW, conv5.dB, y5.buf,
                   conv5.IC, r4.H, r4.W, conv5.OC, conv5.KH, conv5.KW, y5.H, y5.W,
                   conv5.SH, conv5.SW, conv5.PH, conv5.PW,0);
    Tensor r5 = y5; r5.buf = make_device_buffer(ctx, nullptr, r5.bytes(), &err);
    run_relu3d(q, K.k_relu, y5.buf, r5.buf, r5.C, r5.H, r5.W);

    // pool5: 3x3 s=2 p=0 -> 6x6
    Tensor p5( r5.C, OUT_H(r5.H, P1_K, P1_S, P_PAD), OUT_W(r5.W, P1_K, P1_S, P_PAD) );
    p5.buf = make_device_buffer(ctx, nullptr, p5.bytes(), &err);
    run_pool2d(q, K.k_pool, r5.buf, p5.buf, p5.C, r5.H, r5.W, p5.H, p5.W,
           P1_K, P1_K, P1_S, P1_S, P_PAD, P_PAD, 0, 0);

    // ======== Flatten + FC1 -> ReLU -> FC2 -> ReLU -> FC3 ========
    const int FLAT_K = 256 * 6 * 6;

    // FC1
    Tensor fc1_mm(1,1,FC1_SIZE); fc1_mm.buf = make_device_buffer(ctx,nullptr,sizeof(float)*FC1_SIZE,&err);
    run_gemm(q, K.k_gemm, p5.buf, d_fcW1, fc1_mm.buf, 1, FLAT_K, FC1_SIZE, 0, 0);
    Tensor fc1_out(1,1,FC1_SIZE); fc1_out.buf = make_device_buffer(ctx,nullptr,sizeof(float)*FC1_SIZE,&err);
    run_add3d(q, K.k_add, fc1_mm.buf, d_fcB1, fc1_out.buf, 1, 1, FC1_SIZE);
    Tensor fc1_relu = fc1_out; fc1_relu.buf = make_device_buffer(ctx,nullptr,sizeof(float)*FC1_SIZE,&err);
    run_relu3d(q, K.k_relu, fc1_out.buf, fc1_relu.buf, 1, 1, FC1_SIZE);

    // FC2
    Tensor fc2_mm(1,1,FC2_SIZE); fc2_mm.buf = make_device_buffer(ctx,nullptr,sizeof(float)*FC2_SIZE,&err);
    run_gemm(q, K.k_gemm, fc1_relu.buf, d_fcW2, fc2_mm.buf, 1, FC1_SIZE, FC2_SIZE, 0, 0);
    Tensor fc2_out(1,1,FC2_SIZE); fc2_out.buf = make_device_buffer(ctx,nullptr,sizeof(float)*FC2_SIZE,&err);
    run_add3d(q, K.k_add, fc2_mm.buf, d_fcB2, fc2_out.buf, 1, 1, FC2_SIZE);
    Tensor fc2_relu = fc2_out; fc2_relu.buf = make_device_buffer(ctx,nullptr,sizeof(float)*FC2_SIZE,&err);
    run_relu3d(q, K.k_relu, fc2_out.buf, fc2_relu.buf, 1, 1, FC2_SIZE);

    // FC3 (logits)
    Tensor fc3_mm(1,1,NUM_CLASSES); fc3_mm.buf = make_device_buffer(ctx,nullptr,sizeof(float)*NUM_CLASSES,&err);
    run_gemm(q, K.k_gemm, fc2_relu.buf, d_fcW3, fc3_mm.buf, 1, FC2_SIZE, NUM_CLASSES, 0, 0);
    Tensor logits(1,1,NUM_CLASSES); logits.buf = make_device_buffer(ctx,nullptr,sizeof(float)*NUM_CLASSES,&err);
    run_add3d(q, K.k_add, fc3_mm.buf, d_fcB3, logits.buf, 1, 1, NUM_CLASSES);

    clFinish(q);

    // 下载并输出最后一层结果
    std::vector<float> out(NUM_CLASSES);
    download_tensor(q, logits.buf, out.data(), sizeof(float)*NUM_CLASSES);

    // 打印 logits（前 10 个与全部）
    printf("== AlexNet 最后一层 logits 前10个 ==\n");
    for (int i=0;i<10 && i<NUM_CLASSES;++i){
        printf("[%d] %.6f\n", i, out[i]);
    }
    printf("== 全部 logits ==\n");
    for (int i=0;i<NUM_CLASSES;++i){
        printf("%d %.6f\n", i, out[i]);
    }

    // 资源释放
    clReleaseMemObject(x0.buf);
    clReleaseMemObject(y1.buf); clReleaseMemObject(r1.buf); clReleaseMemObject(p1.buf);
    clReleaseMemObject(y2.buf); clReleaseMemObject(r2.buf); clReleaseMemObject(p2.buf);
    clReleaseMemObject(y3.buf); clReleaseMemObject(r3.buf);
    clReleaseMemObject(y4.buf); clReleaseMemObject(r4.buf);
    clReleaseMemObject(y5.buf); clReleaseMemObject(r5.buf); clReleaseMemObject(p5.buf);
    clReleaseMemObject(fc1_out.buf); clReleaseMemObject(fc1_relu.buf);
    clReleaseMemObject(fc2_out.buf); clReleaseMemObject(fc2_relu.buf);
    clReleaseMemObject(logits.buf);

    clReleaseMemObject(conv1.dW); clReleaseMemObject(conv1.dB);
    clReleaseMemObject(conv2.dW); clReleaseMemObject(conv2.dB);
    clReleaseMemObject(conv3.dW); clReleaseMemObject(conv3.dB);
    clReleaseMemObject(conv4.dW); clReleaseMemObject(conv4.dB);
    clReleaseMemObject(conv5.dW); clReleaseMemObject(conv5.dB);

    clReleaseMemObject(d_fcW1); clReleaseMemObject(d_fcB1);
    clReleaseMemObject(d_fcW2); clReleaseMemObject(d_fcB2);
    clReleaseMemObject(d_fcW3); clReleaseMemObject(d_fcB3);

    if (K.k_conv) clReleaseKernel(K.k_conv);
    if (K.k_convbn) clReleaseKernel(K.k_convbn);
    if (K.k_bn) clReleaseKernel(K.k_bn);
    if (K.k_relu) clReleaseKernel(K.k_relu);
    if (K.k_add) clReleaseKernel(K.k_add);
    if (K.k_pool) clReleaseKernel(K.k_pool);
    if (K.k_gemm) clReleaseKernel(K.k_gemm);

    if (K.prog_conv) clReleaseProgram(K.prog_conv);
    if (K.prog_convbn) clReleaseProgram(K.prog_convbn);
    if (K.prog_bn) clReleaseProgram(K.prog_bn);
    if (K.prog_relu) clReleaseProgram(K.prog_relu);
    if (K.prog_add) clReleaseProgram(K.prog_add);
    if (K.prog_pool) clReleaseProgram(K.prog_pool);
    if (K.prog_gemm) clReleaseProgram(K.prog_gemm);

    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    return 0;
}
