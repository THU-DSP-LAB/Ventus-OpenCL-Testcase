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
#include <random>

// ======= Kernel 路径（与现有工程一致）=======
static const char* KPATH_CONV      = "../AIops/conv_fused/conv_fused.cl"; // 修改为卷积+ReLU融合的kernel路径
static const char* KPATH_POOL2D    = "../AIops/Pool2D/pool2d.cl";
static const char* KPATH_GEMM_FUSED = "../AIops/gemm_fused/gemm_fused.cl"; // 融合版本
static const char* KPATH_GEMM_PLAIN = "../AIops/GEMM/gemm.cl"; // 纯GEMM版本

// ======= 配置：LeNet-5（32x32，3通道）=======
static const int  NUM_CLASSES = 10;
static const int  N = 1;                 // Batch 固定 1，布局 NCHW
static const int  IN_C = 3, IN_H = 32, IN_W = 32;

// ======= 固定随机种子工具 =======
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
struct Tensor {
    int C,H,W;             
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
static void zero_buffer(cl_command_queue q, cl_mem buf, size_t bytes) {
    const cl_uint zero = 0;
    cl_int e = clEnqueueFillBuffer(q, buf, &zero, sizeof(zero), 0, bytes, 0, NULL, NULL);
    if (e != CL_SUCCESS) {
        fprintf(stderr, "FillBuffer(0) failed: %d\n", e);
        exit(1);
    }
}
static char* load_text_file(const char* path, size_t* out_size = nullptr){
    FILE* fp = fopen(path, "rb");
    if(!fp){ fprintf(stderr, "无法打开 kernel 文件: %s\n", path); exit(1); }
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); rewind(fp);
    char* buf = (char*)malloc(sz+1);
    fread(buf,1,sz,fp); buf[sz]='\0'; fclose(fp);
    if(out_size) *out_size = (size_t)sz;
    return buf;
}
// 下载数据从 GPU 到主机
static void download_tensor(cl_command_queue q, cl_mem buf, void* dst, size_t bytes) {
    cl_int e = clEnqueueReadBuffer(q, buf, CL_TRUE, 0, bytes, dst, 0, NULL, NULL);
    if (e != CL_SUCCESS) {
        fprintf(stderr, "EnqueueReadBuffer failed: %d\n", e);
        exit(1);
    }
    clFinish(q); // 确保命令队列执行完
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
static cl_mem make_device_buffer(cl_context ctx, const void* host, size_t bytes, cl_int* err=nullptr){
    cl_int e;
    cl_mem b = clCreateBuffer(ctx, CL_MEM_READ_WRITE | (host?CL_MEM_COPY_HOST_PTR:0), bytes, (void*)host, &e);
    if(err) *err=e;
    if(e!=CL_SUCCESS){ fprintf(stderr,"clCreateBuffer failed: %d\n", e); exit(1); }
    return b;
}

// ======= kernel 句柄集合 =======
struct Kernels {
    cl_program prog_conv=nullptr, prog_pool=nullptr, prog_gemm_fused=nullptr, prog_gemm_plain=nullptr;
    cl_kernel  k_conv=nullptr,  k_pool=nullptr,  k_gemm_relu=nullptr, k_gemm_plain=nullptr;
};
static void build_all_kernels(cl_context ctx, cl_device_id dev, Kernels& K){
    cl_int err;
    K.prog_conv   = build_program(ctx, dev, KPATH_CONV);
    K.k_conv      = clCreateKernel(K.prog_conv, "conv2d_relu", &err); // 使用卷积+ReLU融合的kernel
    if (err!=CL_SUCCESS){ fprintf(stderr,"clCreateKernel(conv2d_relu) err=%d\n", err); exit(1); }

    K.prog_pool   = build_program(ctx, dev, KPATH_POOL2D);
    K.k_pool      = clCreateKernel(K.prog_pool, "pool2d", &err);
    if (err!=CL_SUCCESS){ fprintf(stderr,"clCreateKernel(pool2d) err=%d\n", err); exit(1); }

    K.prog_gemm_fused   = build_program(ctx, dev, KPATH_GEMM_FUSED);
    K.k_gemm_relu = clCreateKernel(K.prog_gemm_fused, "gemm_relu", &err); // 使用矩阵乘法+ReLU融合的kernel
    if (err!=CL_SUCCESS){ fprintf(stderr,"clCreateKernel(gemm_relu) err=%d\n", err); exit(1); }

    K.prog_gemm_plain   = build_program(ctx, dev, KPATH_GEMM_PLAIN);
    K.k_gemm_plain = clCreateKernel(K.prog_gemm_plain, "gemm", &err); // 纯矩阵乘法kernel
    if (err!=CL_SUCCESS){ fprintf(stderr,"clCreateKernel(gemm) err=%d\n", err); exit(1); }
}

// ======= 卷积层参数/工具 =======
struct ConvParams {
    int IC, OC, KH, KW, SH, SW, PH, PW;
    std::vector<float> W; // 移除偏置 B
    cl_mem dW = nullptr;
};
static void init_conv_seeded(ConvParams& p, cl_context ctx, uint32_t seed_w) {
    p.W.resize((size_t)p.OC * p.IC * p.KH * p.KW);
    fill_uniform(p.W, -0.1f, 0.1f, seed_w);
    p.dW = make_device_buffer(ctx, p.W.data(), p.W.size() * sizeof(float));
}
static inline int OUT_H(int IH,int KH,int SH,int PH) { return (IH + 2*PH - KH) / SH + 1; }
static inline int OUT_W(int IW,int KW,int SW,int PW) { return (IW + 2*PW - KW) / SW + 1; }

// ======= 全连接层参数/工具 =======
struct GemmParams {
    int in_dim, out_dim;
    std::vector<float> W; // 移除偏置 B
    cl_mem dW = nullptr;
};
static void init_gemm_seeded(GemmParams& p, cl_context ctx, uint32_t seed_w) {
    p.W.resize((size_t)p.in_dim * p.out_dim);
    fill_uniform(p.W, -0.1f, 0.1f, seed_w);
    p.dW = make_device_buffer(ctx, p.W.data(), p.W.size() * sizeof(float));
}

// ======= 核函数调用封装 =======
static void run_conv_layer(cl_command_queue q, cl_kernel k_conv,
                           cl_mem in, cl_mem w, cl_mem out,
                           int IC, int IH, int IW,
                           int OC, int KH, int KW,
                           int OH, int OW, int SH, int SW, int PH, int PW)
{
    int arg = 0;
    clSetKernelArg(k_conv, arg++, sizeof(cl_mem), &in);
    clSetKernelArg(k_conv, arg++, sizeof(cl_mem), &w);
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

    size_t g[3] = {(size_t)OC, (size_t)OH, (size_t)OW};
    cl_int e = clEnqueueNDRangeKernel(q, k_conv, 3, NULL, g, NULL, 0, NULL, NULL);
    if (e!=CL_SUCCESS){ fprintf(stderr,"Enqueue conv failed: %d\n", e); exit(1); }
}

static void run_pool_layer(cl_command_queue q, cl_kernel k_pool,
                           cl_mem in, cl_mem out,
                           int C, int IH, int IW,
                           int KH, int KW, int SH, int SW, int PH, int PW, int OH, int OW,
                           int mode, int count_include_pad)
{
    int arg = 0;
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

static void run_gemm_layer(cl_command_queue q, cl_kernel k_gemm,
                           cl_mem A, cl_mem B, cl_mem C,
                           int M, int K, int N, int do_trans_a, int do_trans_b)
{
    int arg = 0;
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

// ======= Forward: conv1 -> pool(2x2) -> conv2 -> pool(2x2) -> Flatten -> FC layers -> logits =======
int main() {
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

    // 初始化输入
    Tensor x0(IN_C, IN_H, IN_W);
    fill_uniform_tensor(x0, -1.f, 1.f, seed_mix(0x13572468, 1)); // 输入 seed
    x0.buf = make_device_buffer(ctx, x0.host.data(), x0.bytes());

    // conv1 参数与执行
    ConvParams conv1; conv1.IC=IN_C; conv1.OC=6; conv1.KH=5; conv1.KW=5; conv1.SH=1; conv1.SW=1; conv1.PH=2; conv1.PW=2;
    init_conv_seeded(conv1, ctx, seed_mix(0x13572468, 101));
    Tensor y1(conv1.OC, OUT_H(x0.H, conv1.KH, conv1.SH, conv1.PH), OUT_W(x0.W, conv1.KW, conv1.SW, conv1.PW));
    y1.buf = make_device_buffer(ctx, nullptr, y1.bytes());
    zero_buffer(q, y1.buf, y1.bytes());
    run_conv_layer(q, K.k_conv, x0.buf, conv1.dW, y1.buf,
                   conv1.IC, x0.H, x0.W, conv1.OC, conv1.KH, conv1.KW, y1.H, y1.W,
                   conv1.SH, conv1.SW, conv1.PH, conv1.PW);

    // pool1 执行 (假设 kernel 为 average pooling)
    int pool_k = 2, pool_s = 2, pool_p = 0;
    int mode = 1; // avg
    int count_include_pad = 0; // exclude pad
    Tensor y2(y1.C, OUT_H(y1.H, pool_k, pool_s, pool_p), OUT_W(y1.W, pool_k, pool_s, pool_p));
    y2.buf = make_device_buffer(ctx, nullptr, y2.bytes());
    zero_buffer(q, y2.buf, y2.bytes());
    run_pool_layer(q, K.k_pool, y1.buf, y2.buf,
                   y1.C, y1.H, y1.W, pool_k, pool_k, pool_s, pool_s, pool_p, pool_p, y2.H, y2.W,
                   mode, count_include_pad);

    // conv2 参数与执行
    ConvParams conv2; conv2.IC=y2.C; conv2.OC=16; conv2.KH=5; conv2.KW=5; conv2.SH=1; conv2.SW=1; conv2.PH=0; conv2.PW=0;
    init_conv_seeded(conv2, ctx, seed_mix(0x13572468, 102));
    Tensor y3(conv2.OC, OUT_H(y2.H, conv2.KH, conv2.SH, conv2.PH), OUT_W(y2.W, conv2.KW, conv2.SW, conv2.PW));
    y3.buf = make_device_buffer(ctx, nullptr, y3.bytes());
    zero_buffer(q, y3.buf, y3.bytes());
    run_conv_layer(q, K.k_conv, y2.buf, conv2.dW, y3.buf,
                   conv2.IC, y2.H, y2.W, conv2.OC, conv2.KH, conv2.KW, y3.H, y3.W,
                   conv2.SH, conv2.SW, conv2.PH, conv2.PW);

    // pool2 执行
    Tensor y4(y3.C, OUT_H(y3.H, pool_k, pool_s, pool_p), OUT_W(y3.W, pool_k, pool_s, pool_p));
    y4.buf = make_device_buffer(ctx, nullptr, y4.bytes());
    zero_buffer(q, y4.buf, y4.bytes());
    run_pool_layer(q, K.k_pool, y3.buf, y4.buf,
                   y3.C, y3.H, y3.W, pool_k, pool_k, pool_s, pool_s, pool_p, pool_p, y4.H, y4.W,
                   mode, count_include_pad);

    // Flatten (使用 y4.buf 作为向量输入，尺寸 y4.elems() = 16*6*6 = 576)
    int flat_dim = y4.elems();

    // FC1 参数与执行 (with ReLU)
    GemmParams fc1; fc1.in_dim = flat_dim; fc1.out_dim = 120;
    init_gemm_seeded(fc1, ctx, seed_mix(0x13572468, 103));
    Tensor fc1_out(1, 1, fc1.out_dim);
    fc1_out.buf = make_device_buffer(ctx, nullptr, fc1_out.bytes());
    zero_buffer(q, fc1_out.buf, fc1_out.bytes());
    run_gemm_layer(q, K.k_gemm_relu, y4.buf, fc1.dW, fc1_out.buf, 1, fc1.in_dim, fc1.out_dim, 0, 0);

    // FC2 参数与执行 (with ReLU)
    GemmParams fc2; fc2.in_dim = fc1.out_dim; fc2.out_dim = 84;
    init_gemm_seeded(fc2, ctx, seed_mix(0x13572468, 104));
    Tensor fc2_out(1, 1, fc2.out_dim);
    fc2_out.buf = make_device_buffer(ctx, nullptr, fc2_out.bytes());
    zero_buffer(q, fc2_out.buf, fc2_out.bytes());
    run_gemm_layer(q, K.k_gemm_relu, fc1_out.buf, fc2.dW, fc2_out.buf, 1, fc2.in_dim, fc2.out_dim, 0, 0);

    // FC3 参数与执行 (no ReLU)
    GemmParams fc3; fc3.in_dim = fc2.out_dim; fc3.out_dim = NUM_CLASSES;
    init_gemm_seeded(fc3, ctx, seed_mix(0x13572468, 105));
    Tensor fc3_out(1, 1, fc3.out_dim);
    fc3_out.buf = make_device_buffer(ctx, nullptr, fc3_out.bytes());
    zero_buffer(q, fc3_out.buf, fc3_out.bytes());
    run_gemm_layer(q, K.k_gemm_plain, fc2_out.buf, fc3.dW, fc3_out.buf, 1, fc3.in_dim, fc3.out_dim, 0, 0);

    // 调试：确保所有操作完成
    clFinish(q);  // 等待所有命令执行完

    // 下载 logits 输出
    download_tensor(q, fc3_out.buf, fc3_out.host.data(), fc3_out.bytes());
    printf("== LeNet-5 logits (deterministic) ==\n");
    for (int i = 0; i < NUM_CLASSES; ++i) {
        printf("%d: %.6f\n", i, fc3_out.host[i]);
    }

    // 清理资源 (可选，但推荐)
    clReleaseMemObject(x0.buf); clReleaseMemObject(y1.buf); clReleaseMemObject(y2.buf);
    clReleaseMemObject(y3.buf); clReleaseMemObject(y4.buf); clReleaseMemObject(fc1_out.buf);
    clReleaseMemObject(fc2_out.buf); clReleaseMemObject(fc3_out.buf);
    clReleaseMemObject(conv1.dW); clReleaseMemObject(conv2.dW);
    clReleaseMemObject(fc1.dW); clReleaseMemObject(fc2.dW); clReleaseMemObject(fc3.dW);
    clReleaseKernel(K.k_conv); clReleaseKernel(K.k_pool);
    clReleaseKernel(K.k_gemm_relu); clReleaseKernel(K.k_gemm_plain);
    clReleaseProgram(K.prog_conv); clReleaseProgram(K.prog_pool); clReleaseProgram(K.prog_gemm_fused);
    clReleaseProgram(K.prog_gemm_plain);
    clReleaseCommandQueue(q); clReleaseContext(ctx);

    return 0;
}