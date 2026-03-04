#include <CL/cl.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <cassert>
#include <iostream>
#include <random>   // 固定种子用
#include <fstream>
#include <sstream>
#include "../common/ventus_result_io.h"
// ======= Kernel 路径（与现有工程一致）=======
static const char* KPATH_CONV      = "../AIops/Conv/conv.cl";
static const char* KPATH_RELU      = "../AIops/ReLU/relu.cl";
static const char* KPATH_ADD       = "../AIops/add/add.cl";
static const char* KPATH_POOL2D    = "../AIops/Pool2D/pool2d.cl";
static const char* KPATH_GEMM      = "../AIops/GEMM/gemm.cl";

// ======= 配置：LeNet-5（32x32，3通道）=======
static const int  NUM_CLASSES = 10;
static const int  N = 1;                 // Batch 固定 1，布局 NCHW
static const int  IN_C = 3, IN_H = 32, IN_W = 32;
static const double REGRESSION_ATOL = 1e-4;
static const double REGRESSION_RTOL = 1e-3;
static const double REL_EPS = 1e-12;
static const int MAX_MISMATCH_REPORT = 10;

// ======= 固定随机种子工具 =======
// 给每个数组（输入/每层权重/偏置等）分配独立 seed，避免生成顺序改变带来差异
static inline uint32_t seed_mix(uint32_t base, uint32_t tag){
    // 简单可复现的混合哈希
    uint32_t x = base ^ (tag + 0x9e3779b9u + (base<<6) + (base>>2));
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
static void fill_uniform(std::vector<float>& v, float lo, float hi, uint32_t seed){
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto &x : v) x = dist(rng);
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
static void fill_uniform_tensor(Tensor& t, float lo, float hi, uint32_t seed){
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto &x : t.host) x = dist(rng);
}

// ======= 文件/编译/内存工具 =======
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
    // 如需更稳定的浮点，可把第三个参数换成如下 opts（留空也可）
    // const char* opts = "-cl-std=CL1.2 -cl-no-signed-zeros -cl-fp32-correctly-rounded-divide-sqrt";
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
// 关键：将将要被写入/累加的 buffer 置 0（float 的 0.0 与 uint32 位模式相同）
static void zero_buffer(cl_command_queue q, cl_mem buf, size_t bytes){
    const cl_uint zero = 0;
    cl_int e = clEnqueueFillBuffer(q, buf, &zero, sizeof(zero), 0, bytes, 0, NULL, NULL);
    if(e!=CL_SUCCESS){ fprintf(stderr,"FillBuffer(0) failed: %d\n", e); exit(1); }
}

// ======= kernel 句柄集合 =======
struct Kernels {
    cl_program prog_conv=nullptr, prog_relu=nullptr, prog_add=nullptr, prog_pool=nullptr, prog_gemm=nullptr;
    cl_kernel  k_conv=nullptr,  k_relu=nullptr,  k_add=nullptr,  k_pool=nullptr,  k_gemm=nullptr;
};
static void build_all_kernels(cl_context ctx, cl_device_id dev, Kernels& K){
    cl_int err;
    K.prog_conv   = build_program(ctx, dev, KPATH_CONV);
    K.k_conv      = clCreateKernel(K.prog_conv, "conv2d", &err);
    if (err!=CL_SUCCESS){ fprintf(stderr,"clCreateKernel(conv2d) err=%d\n", err); exit(1); }

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

// ======= 卷积层参数/工具 =======
struct ConvParams {
    int IC, OC, KH, KW, SH, SW, PH, PW;
    std::vector<float> W, B;
    cl_mem dW=nullptr, dB=nullptr;
};
static void init_conv_seeded(ConvParams& p, uint32_t seed_w, uint32_t seed_b) {
    p.W.resize((size_t)p.OC * p.IC * p.KH * p.KW);
    p.B.resize(p.OC);
    fill_uniform(p.W, -0.1f, 0.1f, seed_w);
    fill_uniform(p.B, -0.1f, 0.1f, seed_b);
}
static inline int OUT_H(int IH,int KH,int SH,int PH) { return (IH + 2*PH - KH) / SH + 1; }
static inline int OUT_W(int IW,int KW,int SW,int PW) { return (IW + 2*PW - KW) / SW + 1; }

// ======= 核函数调用封装 =======
static void run_conv_layer(cl_command_queue q, cl_kernel k_conv,
                           cl_mem in, cl_mem w, cl_mem b, cl_mem out,
                           int IC, int IH, int IW,
                           int OC, int KH, int KW,
                           int OH, int OW, int SH, int SW, int PH, int PW,
                           int do_relu)
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
    clSetKernelArg(k_conv, arg++, sizeof(int), &do_relu);

    size_t g[3] = {(size_t)OC, (size_t)OH, (size_t)OW};
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
    size_t g[3] = {(size_t)C, (size_t)H, (size_t)W};
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

    size_t g[3] = {(size_t)C, (size_t)OH, (size_t)OW};
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
    size_t g[2] = {(size_t)M, (size_t)N};
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

struct LeNetCpuReferenceInput {
    const Tensor& x0;
    const ConvParams& conv1;
    const ConvParams& conv2;
    const std::vector<float>& fcW1;
    const std::vector<float>& fcB1;
    const std::vector<float>& fcW2;
    const std::vector<float>& fcB2;
    const std::vector<float>& fcW3;
    const std::vector<float>& fcB3;
};

static void expect_size(const std::vector<float>& v, size_t expected, const char* name){
    if (v.size() != expected) {
        fprintf(stderr, "%s size mismatch: got=%zu expect=%zu\n", name, v.size(), expected);
        exit(1);
    }
}

static std::vector<float> cpu_conv2d_reference(
    const std::vector<float>& input,
    int in_c, int in_h, int in_w,
    const ConvParams& p,
    int do_relu)
{
    const int out_h = OUT_H(in_h, p.KH, p.SH, p.PH);
    const int out_w = OUT_W(in_w, p.KW, p.SW, p.PW);
    const int in_hw = in_h * in_w;
    const int out_hw = out_h * out_w;
    std::vector<float> output((size_t)p.OC * out_hw, 0.0f);

    for (int oc = 0; oc < p.OC; ++oc) {
        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox = 0; ox < out_w; ++ox) {
                float sum = p.B.empty() ? 0.0f : p.B[oc];
                for (int ic = 0; ic < in_c; ++ic) {
                    for (int ky = 0; ky < p.KH; ++ky) {
                        const int iy = oy * p.SH - p.PH + ky;
                        if (iy < 0 || iy >= in_h) continue;
                        for (int kx = 0; kx < p.KW; ++kx) {
                            const int ix = ox * p.SW - p.PW + kx;
                            if (ix < 0 || ix >= in_w) continue;
                            const int in_idx = ic * in_hw + iy * in_w + ix;
                            const int w_idx = oc * (in_c * p.KH * p.KW)
                                            + ic * (p.KH * p.KW)
                                            + ky * p.KW + kx;
                            sum += input[in_idx] * p.W[w_idx];
                        }
                    }
                }
                if (do_relu && sum < 0.0f) sum = 0.0f;
                output[oc * out_hw + oy * out_w + ox] = sum;
            }
        }
    }
    return output;
}

static void cpu_relu3d_inplace(std::vector<float>& x){
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

static std::vector<float> cpu_max_pool2d_reference(
    const std::vector<float>& input,
    int C, int IH, int IW,
    int KH, int KW, int SH, int SW, int PH, int PW)
{
    const int OH = OUT_H(IH, KH, SH, PH);
    const int OW = OUT_W(IW, KW, SW, PW);
    const int in_hw = IH * IW;
    const int out_hw = OH * OW;
    std::vector<float> output((size_t)C * out_hw, 0.0f);

    for (int c = 0; c < C; ++c) {
        for (int oy = 0; oy < OH; ++oy) {
            for (int ox = 0; ox < OW; ++ox) {
                float acc = -INFINITY;
                for (int ky = 0; ky < KH; ++ky) {
                    const int iy = oy * SH - PH + ky;
                    for (int kx = 0; kx < KW; ++kx) {
                        const int ix = ox * SW - PW + kx;
                        if (iy < 0 || iy >= IH || ix < 0 || ix >= IW) continue;
                        const int idx = c * in_hw + iy * IW + ix;
                        acc = std::max(acc, input[idx]);
                    }
                }
                output[c * out_hw + oy * OW + ox] = acc;
            }
        }
    }
    return output;
}

static std::vector<float> cpu_gemm_reference(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M, int K, int N,
    int do_trans_a, int do_trans_b)
{
    std::vector<float> C((size_t)M * N, 0.0f);
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                const float a_val = do_trans_a ? A[k * M + row] : A[row * K + k];
                const float b_val = do_trans_b ? B[col * K + k] : B[k * N + col];
                sum += a_val * b_val;
            }
            C[row * N + col] = sum;
        }
    }
    return C;
}

static std::vector<float> cpu_add_reference(
    const std::vector<float>& a,
    const std::vector<float>& b)
{
    if (a.size() != b.size()) {
        fprintf(stderr, "add input size mismatch: a=%zu b=%zu\n", a.size(), b.size());
        exit(1);
    }
    std::vector<float> out(a.size(), 0.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

static std::vector<float> run_lenet5_cpu_reference(const LeNetCpuReferenceInput& in){
    const int P_K = 2;
    const int P_S = 2;
    const int P_PAD = 0;
    const int FC1_SIZE = 120;
    const int FC2_SIZE = 84;
    const int FLAT_K = 16 * 6 * 6;

    const int y1_h = OUT_H(in.x0.H, in.conv1.KH, in.conv1.SH, in.conv1.PH);
    const int y1_w = OUT_W(in.x0.W, in.conv1.KW, in.conv1.SW, in.conv1.PW);
    std::vector<float> y1 = cpu_conv2d_reference(in.x0.host, in.x0.C, in.x0.H, in.x0.W, in.conv1, 0);
    expect_size(y1, (size_t)in.conv1.OC * y1_h * y1_w, "y1");
    cpu_relu3d_inplace(y1);

    const int p1_h = OUT_H(y1_h, P_K, P_S, P_PAD);
    const int p1_w = OUT_W(y1_w, P_K, P_S, P_PAD);
    std::vector<float> p1 = cpu_max_pool2d_reference(y1, in.conv1.OC, y1_h, y1_w, P_K, P_K, P_S, P_S, P_PAD, P_PAD);
    expect_size(p1, (size_t)in.conv1.OC * p1_h * p1_w, "p1");

    const int y2_h = OUT_H(p1_h, in.conv2.KH, in.conv2.SH, in.conv2.PH);
    const int y2_w = OUT_W(p1_w, in.conv2.KW, in.conv2.SW, in.conv2.PW);
    std::vector<float> y2 = cpu_conv2d_reference(p1, in.conv2.IC, p1_h, p1_w, in.conv2, 0);
    expect_size(y2, (size_t)in.conv2.OC * y2_h * y2_w, "y2");
    cpu_relu3d_inplace(y2);

    const int p2_h = OUT_H(y2_h, P_K, P_S, P_PAD);
    const int p2_w = OUT_W(y2_w, P_K, P_S, P_PAD);
    std::vector<float> p2 = cpu_max_pool2d_reference(y2, in.conv2.OC, y2_h, y2_w, P_K, P_K, P_S, P_S, P_PAD, P_PAD);
    expect_size(p2, (size_t)in.conv2.OC * p2_h * p2_w, "p2");

    expect_size(p2, FLAT_K, "flatten");
    expect_size(in.fcW1, (size_t)FC1_SIZE * FLAT_K, "fcW1");
    expect_size(in.fcB1, FC1_SIZE, "fcB1");
    expect_size(in.fcW2, (size_t)FC2_SIZE * FC1_SIZE, "fcW2");
    expect_size(in.fcB2, FC2_SIZE, "fcB2");
    expect_size(in.fcW3, (size_t)NUM_CLASSES * FC2_SIZE, "fcW3");
    expect_size(in.fcB3, NUM_CLASSES, "fcB3");

    std::vector<float> fc1 = cpu_add_reference(cpu_gemm_reference(p2, in.fcW1, 1, FLAT_K, FC1_SIZE, 0, 0), in.fcB1);
    cpu_relu3d_inplace(fc1);
    std::vector<float> fc2 = cpu_add_reference(cpu_gemm_reference(fc1, in.fcW2, 1, FC1_SIZE, FC2_SIZE, 0, 0), in.fcB2);
    cpu_relu3d_inplace(fc2);
    return cpu_add_reference(cpu_gemm_reference(fc2, in.fcW3, 1, FC2_SIZE, NUM_CLASSES, 0, 0), in.fcB3);
}

static void print_logits(const char* title, const std::vector<float>& logits){
    printf("== %s ==\n", title);
    for (size_t i = 0; i < logits.size(); ++i) {
        printf("%zu %.6f\n", i, logits[i]);
    }
}

static bool maybe_inject_regression_error(std::vector<float>& out){
    const char* idx_str = std::getenv("LENET5_INJECT_ERR_IDX");
    if (!idx_str || !*idx_str) return false;

    char* end_idx = nullptr;
    const long idx = std::strtol(idx_str, &end_idx, 10);
    if (end_idx == idx_str || *end_idx != '\0' || idx < 0 || (size_t)idx >= out.size()) {
        fprintf(stderr, "Invalid LENET5_INJECT_ERR_IDX=%s, valid range=[0,%zu)\n", idx_str, out.size());
        exit(1);
    }

    float delta = 1.0f;
    const char* delta_str = std::getenv("LENET5_INJECT_ERR_DELTA");
    if (delta_str && *delta_str) {
        char* end_delta = nullptr;
        const float parsed = std::strtof(delta_str, &end_delta);
        if (end_delta != delta_str && *end_delta == '\0') {
            delta = parsed;
        } else {
            fprintf(stderr, "Invalid LENET5_INJECT_ERR_DELTA=%s\n", delta_str);
            exit(1);
        }
    }

    const float before = out[(size_t)idx];
    out[(size_t)idx] = before + delta;
    fprintf(stderr,
            "[DEBUG] Injected regression error at idx=%ld: %.9f -> %.9f (delta=%.9f)\n",
            idx, before, out[(size_t)idx], delta);
    return true;
}

static bool check_regression(
    const std::vector<float>& opencl_out,
    const std::vector<float>& cpu_ref,
    double atol,
    double rtol)
{
    const size_t n = std::min(opencl_out.size(), cpu_ref.size());
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    size_t worst_idx = 0;
    std::vector<size_t> mismatch_indices;

    for (size_t i = 0; i < n; ++i) {
        const double abs_err = std::fabs((double)opencl_out[i] - (double)cpu_ref[i]);
        const double bound = atol + rtol * std::fabs((double)cpu_ref[i]);
        const double rel_err = abs_err / (std::fabs((double)cpu_ref[i]) + REL_EPS);
        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            max_rel_err = rel_err;
            worst_idx = i;
        }
        if (abs_err > bound) {
            mismatch_indices.push_back(i);
        }
    }

    const bool size_match = opencl_out.size() == cpu_ref.size();
    const size_t size_mismatch_cnt = size_match ? 0 : (std::max(opencl_out.size(), cpu_ref.size()) - n);
    const size_t bad = mismatch_indices.size() + size_mismatch_cnt;

    printf("Regression tolerance: atol=%.3e rtol=%.3e\n", atol, rtol);
    printf("Compared elements: %zu, max_abs_err=%.6e, max_rel_err=%.6e (idx=%zu)\n",
           n, max_abs_err, max_rel_err, worst_idx);

    if (bad == 0) {
        printf("Regression check PASSED\n");
        return true;
    }

    fprintf(stderr,
            "\n[REGRESSION][FAIL] opencl_size=%zu cpu_ref_size=%zu mismatches=%zu size_mismatch=%zu\n",
            opencl_out.size(), cpu_ref.size(), mismatch_indices.size(), size_mismatch_cnt);
    fprintf(stderr, " idx |    OpenCL    |   CPU_REF    |   ABS_ERR    |   REL_ERR    |    BOUND\n");
    fprintf(stderr, "-----+--------------+--------------+--------------+--------------+--------------\n");

    const size_t report_n = std::min(mismatch_indices.size(), (size_t)MAX_MISMATCH_REPORT);
    for (size_t j = 0; j < report_n; ++j) {
        const size_t i = mismatch_indices[j];
        const double abs_err = std::fabs((double)opencl_out[i] - (double)cpu_ref[i]);
        const double rel_err = abs_err / (std::fabs((double)cpu_ref[i]) + REL_EPS);
        const double bound = atol + rtol * std::fabs((double)cpu_ref[i]);
        fprintf(stderr, "%4zu | %12.6f | %12.6f | %12.6e | %12.6e | %12.6e\n",
                i, opencl_out[i], cpu_ref[i], abs_err, rel_err, bound);
    }
    if (mismatch_indices.size() > report_n) {
        fprintf(stderr, "... %zu more mismatches not shown\n", mismatch_indices.size() - report_n);
    }
    if (!size_match) {
        if (opencl_out.size() > n) {
            fprintf(stderr, "OpenCL extra outputs start at idx=%zu, count=%zu\n", n, opencl_out.size() - n);
        }
        if (cpu_ref.size() > n) {
            fprintf(stderr, "CPU ref extra outputs start at idx=%zu, count=%zu\n", n, cpu_ref.size() - n);
        }
    }
    return false;
}

int main() {
    // ==== 固定随机种子基值（可改） ====
    const uint32_t SEED_BASE = 0x13572468u;
    int exit_code = 0;

    // OpenCL setup
    cl_int err = CL_SUCCESS;
    cl_platform_id plat; cl_device_id dev;
    err = clGetPlatformIDs(1, &plat, NULL);
    if (err!=CL_SUCCESS){ fprintf(stderr,"clGetPlatformIDs err=%d\n", err); return -1; }
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err!=CL_SUCCESS){
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

    // ===== 输入：固定种子填充 [-1,1] =====
    Tensor x0 = Tensor(IN_C, IN_H, IN_W);
    fill_uniform_tensor(x0, -1.f, 1.f, seed_mix(SEED_BASE, 1));          // 输入 seed
    x0.buf = make_device_buffer(ctx, x0.host.data(), x0.bytes(), &err);

    // ===== LeNet-5 卷积块 =====
    // conv1: 5x5 s=1 p=2, IN_C->6
    ConvParams conv1; conv1.IC=IN_C; conv1.OC=6; conv1.KH=5; conv1.KW=5; conv1.SH=1; conv1.SW=1; conv1.PH=2; conv1.PW=2;
    init_conv_seeded(conv1, seed_mix(SEED_BASE, 101), seed_mix(SEED_BASE, 102));
    // conv2: 5x5 s=1 p=0, 6->16
    ConvParams conv2; conv2.IC=6;    conv2.OC=16; conv2.KH=5; conv2.KW=5; conv2.SH=1; conv2.SW=1; conv2.PH=0; conv2.PW=0;
    init_conv_seeded(conv2, seed_mix(SEED_BASE, 201), seed_mix(SEED_BASE, 202));

    // 上传权重
    conv1.dW = make_device_buffer(ctx, conv1.W.data(), sizeof(float)*conv1.W.size(), &err);
    conv1.dB = make_device_buffer(ctx, conv1.B.data(), sizeof(float)*conv1.B.size(), &err);
    conv2.dW = make_device_buffer(ctx, conv2.W.data(), sizeof(float)*conv2.W.size(), &err);
    conv2.dB = make_device_buffer(ctx, conv2.B.data(), sizeof(float)*conv2.B.size(), &err);

    // ===== Forward: conv1 -> ReLU -> pool(2x2) =====
    Tensor y1( conv1.OC, OUT_H(x0.H, conv1.KH, conv1.SH, conv1.PH), OUT_W(x0.W, conv1.KW, conv1.SW, conv1.PW) );
    y1.buf = make_device_buffer(ctx, nullptr, y1.bytes(), &err);
    zero_buffer(q, y1.buf, y1.bytes());

    run_conv_layer(q, K.k_conv, x0.buf, conv1.dW, conv1.dB, y1.buf,
                   conv1.IC, x0.H, x0.W, conv1.OC, conv1.KH, conv1.KW, y1.H, y1.W,
                   conv1.SH, conv1.SW, conv1.PH, conv1.PW, 0);

    Tensor r1 = y1; r1.buf = make_device_buffer(ctx, nullptr, r1.bytes(), &err);
    zero_buffer(q, r1.buf, r1.bytes());
    run_relu3d(q, K.k_relu, y1.buf, r1.buf, r1.C, r1.H, r1.W);

    // pool1: 2x2 s=2 p=0
    const int P_K=2, P_S=2, P_PAD=0;
    Tensor p1( r1.C, OUT_H(r1.H, P_K, P_S, P_PAD), OUT_W(r1.W, P_K, P_S, P_PAD) );
    p1.buf = make_device_buffer(ctx, nullptr, p1.bytes(), &err);
    zero_buffer(q, p1.buf, p1.bytes());
    run_pool2d(q, K.k_pool, r1.buf, p1.buf, p1.C, r1.H, r1.W, p1.H, p1.W,
               P_K, P_K, P_S, P_S, P_PAD, P_PAD, 0, 0);

    // ===== conv2 -> ReLU -> pool(2x2) =====
    Tensor y2( conv2.OC, OUT_H(p1.H, conv2.KH, conv2.SH, conv2.PH), OUT_W(p1.W, conv2.KW, conv2.SW, conv2.PW) );
    y2.buf = make_device_buffer(ctx, nullptr, y2.bytes(), &err);
    zero_buffer(q, y2.buf, y2.bytes());
    run_conv_layer(q, K.k_conv, p1.buf, conv2.dW, conv2.dB, y2.buf,
                   conv2.IC, p1.H, p1.W, conv2.OC, conv2.KH, conv2.KW, y2.H, y2.W,
                   conv2.SH, conv2.SW, conv2.PH, conv2.PW, 0);

    Tensor r2 = y2; r2.buf = make_device_buffer(ctx, nullptr, r2.bytes(), &err);
    zero_buffer(q, r2.buf, r2.bytes());
    run_relu3d(q, K.k_relu, y2.buf, r2.buf, r2.C, r2.H, r2.W);

    Tensor p2( r2.C, OUT_H(r2.H, P_K, P_S, P_PAD), OUT_W(r2.W, P_K, P_S, P_PAD) );
    p2.buf = make_device_buffer(ctx, nullptr, p2.bytes(), &err);
    zero_buffer(q, p2.buf, p2.bytes());
    run_pool2d(q, K.k_pool, r2.buf, p2.buf, p2.C, r2.H, r2.W, p2.H, p2.W,
               P_K, P_K, P_S, P_S, P_PAD, P_PAD, 0, 0);

    // ===== Flatten + FCs =====
    const int FLAT_K = 16 * 6 * 6;    // = 576
    const int FC1_SIZE = 120;
    const int FC2_SIZE = 84;

    // 初始化 FC 权重（固定种子，范围[-0.1,0.1]）
    std::vector<float> fcW1(FC1_SIZE * FLAT_K), fcB1(FC1_SIZE);
    std::vector<float> fcW2(FC2_SIZE * FC1_SIZE), fcB2(FC2_SIZE);
    std::vector<float> fcW3(NUM_CLASSES * FC2_SIZE), fcB3(NUM_CLASSES);
    fill_uniform(fcW1, -0.1f, 0.1f, seed_mix(SEED_BASE, 301));
    fill_uniform(fcB1, -0.1f, 0.1f, seed_mix(SEED_BASE, 302));
    fill_uniform(fcW2, -0.1f, 0.1f, seed_mix(SEED_BASE, 401));
    fill_uniform(fcB2, -0.1f, 0.1f, seed_mix(SEED_BASE, 402));
    fill_uniform(fcW3, -0.1f, 0.1f, seed_mix(SEED_BASE, 501));
    fill_uniform(fcB3, -0.1f, 0.1f, seed_mix(SEED_BASE, 502));

    const LeNetCpuReferenceInput cpu_ref_input = {
        x0, conv1, conv2, fcW1, fcB1, fcW2, fcB2, fcW3, fcB3
    };
    std::vector<float> cpu_ref_logits = run_lenet5_cpu_reference(cpu_ref_input);

    cl_mem d_fcW1 = make_device_buffer(ctx, fcW1.data(), sizeof(float) * fcW1.size(), &err);
    cl_mem d_fcB1 = make_device_buffer(ctx, fcB1.data(), sizeof(float) * fcB1.size(), &err);
    cl_mem d_fcW2 = make_device_buffer(ctx, fcW2.data(), sizeof(float) * fcW2.size(), &err);
    cl_mem d_fcB2 = make_device_buffer(ctx, fcB2.data(), sizeof(float) * fcB2.size(), &err);
    cl_mem d_fcW3 = make_device_buffer(ctx, fcW3.data(), sizeof(float) * fcW3.size(), &err);
    cl_mem d_fcB3 = make_device_buffer(ctx, fcB3.data(), sizeof(float) * fcB3.size(), &err);

    // FC1: C = A(1x576) * W(576x120)
    Tensor fc1_mm(1,1,FC1_SIZE); fc1_mm.buf = make_device_buffer(ctx,nullptr,sizeof(float)*FC1_SIZE,&err);
    zero_buffer(q, fc1_mm.buf, sizeof(float)*FC1_SIZE);
    run_gemm(q, K.k_gemm, p2.buf, d_fcW1, fc1_mm.buf, 1, FLAT_K, FC1_SIZE, 0, 0);

    Tensor fc1_out(1,1,FC1_SIZE); fc1_out.buf = make_device_buffer(ctx,nullptr,sizeof(float)*FC1_SIZE,&err);
    zero_buffer(q, fc1_out.buf, sizeof(float)*FC1_SIZE);
    run_add3d(q, K.k_add, fc1_mm.buf, d_fcB1, fc1_out.buf, 1, 1, FC1_SIZE);

    Tensor fc1_relu = fc1_out; fc1_relu.buf = make_device_buffer(ctx,nullptr,sizeof(float)*FC1_SIZE,&err);
    zero_buffer(q, fc1_relu.buf, sizeof(float)*FC1_SIZE);
    run_relu3d(q, K.k_relu, fc1_out.buf, fc1_relu.buf, 1, 1, FC1_SIZE);

    // FC2
    Tensor fc2_mm(1,1,FC2_SIZE); fc2_mm.buf = make_device_buffer(ctx,nullptr,sizeof(float)*FC2_SIZE,&err);
    zero_buffer(q, fc2_mm.buf, sizeof(float)*FC2_SIZE);
    run_gemm(q, K.k_gemm, fc1_relu.buf, d_fcW2, fc2_mm.buf, 1, FC1_SIZE, FC2_SIZE, 0, 0);

    Tensor fc2_out(1,1,FC2_SIZE); fc2_out.buf = make_device_buffer(ctx,nullptr,sizeof(float)*FC2_SIZE,&err);
    zero_buffer(q, fc2_out.buf, sizeof(float)*FC2_SIZE);
    run_add3d(q, K.k_add, fc2_mm.buf, d_fcB2, fc2_out.buf, 1, 1, FC2_SIZE);

    Tensor fc2_relu = fc2_out; fc2_relu.buf = make_device_buffer(ctx,nullptr,sizeof(float)*FC2_SIZE,&err);
    zero_buffer(q, fc2_relu.buf, sizeof(float)*FC2_SIZE);
    run_relu3d(q, K.k_relu, fc2_out.buf, fc2_relu.buf, 1, 1, FC2_SIZE);

    // FC3 (logits)
    Tensor fc3_mm(1,1,NUM_CLASSES); fc3_mm.buf = make_device_buffer(ctx,nullptr,sizeof(float)*NUM_CLASSES,&err);
    zero_buffer(q, fc3_mm.buf, sizeof(float)*NUM_CLASSES);
    run_gemm(q, K.k_gemm, fc2_relu.buf, d_fcW3, fc3_mm.buf, 1, FC2_SIZE, NUM_CLASSES, 0, 0);

    Tensor logits(1,1,NUM_CLASSES); logits.buf = make_device_buffer(ctx,nullptr,sizeof(float)*NUM_CLASSES,&err);
    zero_buffer(q, logits.buf, sizeof(float)*NUM_CLASSES);
    run_add3d(q, K.k_add, fc3_mm.buf, d_fcB3, logits.buf, 1, 1, NUM_CLASSES);

    clFinish(q);

    // 下载并输出
    std::vector<float> out(NUM_CLASSES);
    download_tensor(q, logits.buf, out.data(), sizeof(float)*NUM_CLASSES);
    maybe_inject_regression_error(out);

    print_logits("LeNet-5 CPU reference logits", cpu_ref_logits);
    print_logits("LeNet-5 OpenCL logits", out);
    fflush(stdout);
    if (!check_regression(out, cpu_ref_logits, REGRESSION_ATOL, REGRESSION_RTOL)) {
        exit_code = 1;
    }
    ventus_write_final_hex(out);

    // 资源释放
    clReleaseMemObject(x0.buf);
    clReleaseMemObject(y1.buf); clReleaseMemObject(r1.buf); clReleaseMemObject(p1.buf);
    clReleaseMemObject(y2.buf); clReleaseMemObject(r2.buf); clReleaseMemObject(p2.buf);
    clReleaseMemObject(fc1_out.buf); clReleaseMemObject(fc1_relu.buf);
    clReleaseMemObject(fc2_out.buf); clReleaseMemObject(fc2_relu.buf);
    clReleaseMemObject(fc1_mm.buf);  clReleaseMemObject(fc2_mm.buf);
    clReleaseMemObject(fc3_mm.buf);  clReleaseMemObject(logits.buf);

    clReleaseMemObject(conv1.dW); clReleaseMemObject(conv1.dB);
    clReleaseMemObject(conv2.dW); clReleaseMemObject(conv2.dB);

    clReleaseMemObject(d_fcW1); clReleaseMemObject(d_fcB1);
    clReleaseMemObject(d_fcW2); clReleaseMemObject(d_fcB2);
    clReleaseMemObject(d_fcW3); clReleaseMemObject(d_fcB3);

    if (K.k_conv) clReleaseKernel(K.k_conv);
    if (K.k_relu) clReleaseKernel(K.k_relu);
    if (K.k_add)  clReleaseKernel(K.k_add);
    if (K.k_pool) clReleaseKernel(K.k_pool);
    if (K.k_gemm) clReleaseKernel(K.k_gemm);

    if (K.prog_conv)  clReleaseProgram(K.prog_conv);
    if (K.prog_relu)  clReleaseProgram(K.prog_relu);
    if (K.prog_add)   clReleaseProgram(K.prog_add);
    if (K.prog_pool)  clReleaseProgram(K.prog_pool);
    if (K.prog_gemm)  clReleaseProgram(K.prog_gemm);

    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return exit_code;
}
