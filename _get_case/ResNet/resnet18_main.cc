// resnet18_main.cc — Host side to run ResNet-18 forward with OpenCL kernels
// 按给定目录结构分别加载各算子 .cl 文件；随机初始化权重；执行 ResNet-18 前向；输出 logits[0..9]
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <cassert>

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
    clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, bytes, src, 0, NULL, NULL);
}
static void download_tensor(cl_command_queue q, cl_mem buf, void* dst, size_t bytes){
    clEnqueueReadBuffer(q, buf, CL_TRUE, 0, bytes, dst, 0, NULL, NULL);
}

// ======= kernel 句柄集合 =======
struct Kernels {
    cl_program prog_conv=nullptr, prog_convbn=nullptr, prog_bn=nullptr;
    cl_program prog_relu=nullptr, prog_add=nullptr, prog_pool=nullptr, prog_gemm=nullptr;

    cl_kernel  k_conv=nullptr, k_convbn=nullptr, k_bn=nullptr;
    cl_kernel  k_relu=nullptr, k_add=nullptr, k_pool=nullptr, k_gemm=nullptr;
};

static void build_all_kernels(cl_context ctx, cl_device_id dev, Kernels& K){
    if (USE_FUSED_CONV_BN) {
        K.prog_convbn = build_program(ctx, dev, KPATH_CONV_BN);
        K.k_convbn    = clCreateKernel(K.prog_convbn, "conv2d_bn", NULL);
    } else {
        K.prog_conv   = build_program(ctx, dev, KPATH_CONV);
        K.k_conv      = clCreateKernel(K.prog_conv, "conv", NULL);
        K.prog_bn     = build_program(ctx, dev, KPATH_BN);
        K.k_bn        = clCreateKernel(K.prog_bn, "batchnorm2d_infer", NULL);
    }
    K.prog_relu   = build_program(ctx, dev, KPATH_RELU);
    K.k_relu      = clCreateKernel(K.prog_relu, "relu3d", NULL);

    K.prog_add    = build_program(ctx, dev, KPATH_ADD);
    K.k_add       = clCreateKernel(K.prog_add, "add3d", NULL);

    K.prog_pool   = build_program(ctx, dev, KPATH_POOL2D);
    K.k_pool      = clCreateKernel(K.prog_pool, "pool2d", NULL);

    K.prog_gemm   = build_program(ctx, dev, KPATH_GEMM);
    K.k_gemm      = clCreateKernel(K.prog_gemm, "gemm", NULL);
}

// ======= 算子封装（host 调用）=======

// Conv_BN：  output[OC,OH,OW] = BN( Conv(input[IC,IH,IW]) )
static void run_conv_bn(cl_command_queue q, cl_kernel k,
                        cl_mem in, cl_mem w, cl_mem b, cl_mem gamma, cl_mem beta, cl_mem mean, cl_mem var,
                        float eps,
                        cl_mem out,
                        int IC,int IH,int IW, int OC,int KH,int KW, int OH,int OW, int SH,int SW, int PH,int PW)
{
    int arg=0;
    clSetKernelArg(k,arg++,sizeof(cl_mem),&in);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&w);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&b);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&gamma);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&beta);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&mean);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&var);
    clSetKernelArg(k,arg++,sizeof(float),&eps);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&out);
    clSetKernelArg(k,arg++,sizeof(int),&IC);
    clSetKernelArg(k,arg++,sizeof(int),&IH);
    clSetKernelArg(k,arg++,sizeof(int),&IW);
    clSetKernelArg(k,arg++,sizeof(int),&OC);
    clSetKernelArg(k,arg++,sizeof(int),&KH);
    clSetKernelArg(k,arg++,sizeof(int),&KW);
    clSetKernelArg(k,arg++,sizeof(int),&OH);
    clSetKernelArg(k,arg++,sizeof(int),&OW);
    clSetKernelArg(k,arg++,sizeof(int),&SH);
    clSetKernelArg(k,arg++,sizeof(int),&SW);
    clSetKernelArg(k,arg++,sizeof(int),&PH);
    clSetKernelArg(k,arg++,sizeof(int),&PW);

    size_t g[3]={(size_t)OC,(size_t)OH,(size_t)OW};
    clEnqueueNDRangeKernel(q, k, 3, NULL, g, NULL, 0, NULL, NULL);
}

// Conv（不融合 BN）
static void run_conv(cl_command_queue q, cl_kernel k,
                     cl_mem in, cl_mem w, cl_mem b, cl_mem out,
                     int IC,int IH,int IW, int KH,int KW, int OH,int OW, int do_relu, int SH,int SW)
{
    int arg=0;
    clSetKernelArg(k,arg++,sizeof(cl_mem),&in);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&w);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&b);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&out);
    clSetKernelArg(k,arg++,sizeof(int),&IC);
    clSetKernelArg(k,arg++,sizeof(int),&IH);
    clSetKernelArg(k,arg++,sizeof(int),&IW);
    clSetKernelArg(k,arg++,sizeof(int),&KH);
    clSetKernelArg(k,arg++,sizeof(int),&KW);
    clSetKernelArg(k,arg++,sizeof(int),&OH);
    clSetKernelArg(k,arg++,sizeof(int),&OW);
    clSetKernelArg(k,arg++,sizeof(int),&do_relu);
    clSetKernelArg(k,arg++,sizeof(int),&SH);
    clSetKernelArg(k,arg++,sizeof(int),&SW);

    size_t g[3]={(size_t)(/*out_c*/0), (size_t)OH,(size_t)OW}; // 注意：你的 conv.cl 里 global 是 {out_channels, out_h, out_w}
    // 由于这个 run_conv 没有 out_c 入参（因各实现不同），如果你的 conv.cl 需要 out_c，记得改 kernel 参数 & 这里的 g[0]
    // 这里仅示意，推荐统一用 conv_bn 跑推理。
    (void)g; // 避免未使用警告
}

// BN（推理）
static void run_bn_infer(cl_command_queue q, cl_kernel k,
                         cl_mem x, cl_mem gamma, cl_mem beta, cl_mem mean, cl_mem var, float eps,
                         cl_mem y, int C,int H,int W)
{
    int arg=0;
    clSetKernelArg(k,arg++,sizeof(cl_mem),&x);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&gamma);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&beta);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&mean);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&var);
    clSetKernelArg(k,arg++,sizeof(float),&eps);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&y);
    clSetKernelArg(k,arg++,sizeof(int),&C);
    clSetKernelArg(k,arg++,sizeof(int),&H);
    clSetKernelArg(k,arg++,sizeof(int),&W);
    size_t g[3]={(size_t)C,(size_t)H,(size_t)W};
    clEnqueueNDRangeKernel(q, k, 3, NULL, g, NULL, 0, NULL, NULL);
}

// ReLU
static void run_relu(cl_command_queue q, cl_kernel k, cl_mem x, cl_mem y, int C,int H,int W){
    int arg=0;
    clSetKernelArg(k,arg++,sizeof(cl_mem),&x);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&y);
    clSetKernelArg(k,arg++,sizeof(int),&C);
    clSetKernelArg(k,arg++,sizeof(int),&H);
    clSetKernelArg(k,arg++,sizeof(int),&W);
    size_t g[3]={(size_t)C,(size_t)H,(size_t)W};
    clEnqueueNDRangeKernel(q, k, 3, NULL, g, NULL, 0, NULL, NULL);
}

// Add
static void run_add(cl_command_queue q, cl_kernel k, cl_mem a, cl_mem b, cl_mem out, int C,int H,int W){
    int arg=0;
    clSetKernelArg(k,arg++,sizeof(cl_mem),&a);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&b);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&out);
    clSetKernelArg(k,arg++,sizeof(int),&C);
    clSetKernelArg(k,arg++,sizeof(int),&H);
    clSetKernelArg(k,arg++,sizeof(int),&W);
    size_t g[3]={(size_t)C,(size_t)H,(size_t)W};
    clEnqueueNDRangeKernel(q, k, 3, NULL, g, NULL, 0, NULL, NULL);
}

// Pool2D：mode=0(max),1(avg)；count_include_pad: 0/1
static void run_pool2d(cl_command_queue q, cl_kernel k, cl_mem x, cl_mem y,
                       int C,int IH,int IW,int OH,int OW,int KH,int KW,int SH,int SW,int PH,int PW,int mode,int count_include_pad)
{
    int arg=0;
    clSetKernelArg(k,arg++,sizeof(cl_mem),&x);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&y);
    clSetKernelArg(k,arg++,sizeof(int),&C);
    clSetKernelArg(k,arg++,sizeof(int),&IH);
    clSetKernelArg(k,arg++,sizeof(int),&IW);
    clSetKernelArg(k,arg++,sizeof(int),&OH);
    clSetKernelArg(k,arg++,sizeof(int),&OW);
    clSetKernelArg(k,arg++,sizeof(int),&KH);
    clSetKernelArg(k,arg++,sizeof(int),&KW);
    clSetKernelArg(k,arg++,sizeof(int),&SH);
    clSetKernelArg(k,arg++,sizeof(int),&SW);
    clSetKernelArg(k,arg++,sizeof(int),&PH);
    clSetKernelArg(k,arg++,sizeof(int),&PW);
    clSetKernelArg(k,arg++,sizeof(int),&mode);
    clSetKernelArg(k,arg++,sizeof(int),&count_include_pad);
    size_t g[3]={(size_t)C,(size_t)OH,(size_t)OW};
    clEnqueueNDRangeKernel(q, k, 3, NULL, g, NULL, 0, NULL, NULL);
}

// GEMM：C[MxN] = A[MxK] * B[KxN]（与现有 gemm.cl 一致）
static void run_gemm(cl_command_queue q, cl_kernel k,
                     cl_mem A, cl_mem B, cl_mem C,
                     int M,int K,int N, int transA, int transB)
{
    int arg=0;
    clSetKernelArg(k,arg++,sizeof(cl_mem),&A);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&B);
    clSetKernelArg(k,arg++,sizeof(cl_mem),&C);
    clSetKernelArg(k,arg++,sizeof(int),&M);
    clSetKernelArg(k,arg++,sizeof(int),&K);
    clSetKernelArg(k,arg++,sizeof(int),&N);
    clSetKernelArg(k,arg++,sizeof(int),&transA);
    clSetKernelArg(k,arg++,sizeof(int),&transB);
    size_t g[2]={(size_t)M,(size_t)N};
    clEnqueueNDRangeKernel(q, k, 2, NULL, g, NULL, 0, NULL, NULL);
}

// ========== ResNet18 结构帮助 ==========
struct ConvBNParams {
    // weights: [OC,IC,KH,KW], bias:[OC], BN: gamma,beta,mean,var [OC]
    int IC,OC,KH,KW,SH,SW,PH,PW;
    std::vector<float> W, B, gamma, beta, mean, var;
    cl_mem dW=nullptr,dB=nullptr,dG=nullptr,dBt=nullptr,dMean=nullptr,dVar=nullptr;
};

// 计算输出尺寸
static inline int OUT_H(int IH,int KH,int SH,int PH){ return (IH + 2*PH - KH)/SH + 1; }
static inline int OUT_W(int IW,int KW,int SW,int PW){ return (IW + 2*PW - KW)/SW + 1; }

// 随机初始化 Conv+BN
static void init_convbn(ConvBNParams& p){
    p.W.resize((size_t)p.OC * p.IC * p.KH * p.KW);
    p.B.resize(p.OC);
    p.gamma.resize(p.OC);
    p.beta.resize(p.OC);
    p.mean.resize(p.OC);
    p.var.resize(p.OC);
    for (auto &v : p.W) v = frand()*0.1f;
    for (int i=0;i<p.OC;++i){
        p.B[i]     = frand()*0.1f;
        p.gamma[i] = frand()*0.1f + 1.0f;
        p.beta[i]  = frand()*0.1f;
        p.mean[i]  = frand()*0.1f;
        p.var[i]   = fabsf(frand())*0.1f + 0.9f; // 正数
    }
}

// 上传到设备（返回带 device 缓冲区的拷贝）
static void upload_convbn(cl_context ctx, const ConvBNParams& p, ConvBNParams& d, cl_int* err=nullptr){
    d = p; // 拷贝形状与 host 向量（host 向量没用到也无妨）
    d.dW    = make_device_buffer(ctx, p.W.data(),     sizeof(float)*p.W.size(),     err);
    d.dB    = make_device_buffer(ctx, p.B.data(),     sizeof(float)*p.B.size(),     err);
    d.dG    = make_device_buffer(ctx, p.gamma.data(), sizeof(float)*p.gamma.size(), err);
    d.dBt   = make_device_buffer(ctx, p.beta.data(),  sizeof(float)*p.beta.size(),  err);
    d.dMean = make_device_buffer(ctx, p.mean.data(),  sizeof(float)*p.mean.size(),  err);
    d.dVar  = make_device_buffer(ctx, p.var.data(),   sizeof(float)*p.var.size(),   err);
}

static Tensor make_device_tensor(cl_context ctx, int C,int H,int W){
    Tensor t(C,H,W);
    for(size_t i=0;i<t.elems();++i) t.host[i]=frand();
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
        run_conv_bn(q, K.k_convbn, x.buf,
                    conv1.dW, conv1.dB, conv1.dG, conv1.dBt, conv1.dMean, conv1.dVar, BN_EPS, y1.buf,
                    conv1.IC, x.H, x.W, conv1.OC, conv1.KH, conv1.KW, oh1, ow1, conv1.SH, conv1.SW, conv1.PH, conv1.PW);

        Tensor y1r = empty_device_tensor(ctx, y1.C, y1.H, y1.W);
        run_relu(q, K.k_relu, y1.buf, y1r.buf, y1.C, y1.H, y1.W);

        // conv2
        int oh2 = OUT_H(y1r.H, conv2.KH, conv2.SH, conv2.PH);
        int ow2 = OUT_W(y1r.W, conv2.KW, conv2.SW, conv2.PW);
        Tensor y2 = empty_device_tensor(ctx, conv2.OC, oh2, ow2);
        run_conv_bn(q, K.k_convbn, y1r.buf,
                    conv2.dW, conv2.dB, conv2.dG, conv2.dBt, conv2.dMean, conv2.dVar, BN_EPS, y2.buf,
                    conv2.IC, y1r.H, y1r.W, conv2.OC, conv2.KH, conv2.KW, oh2, ow2, conv2.SH, conv2.SW, conv2.PH, conv2.PW);

        // identity（下采样？）
        Tensor id;
        if (has_down){
            int odh = OUT_H(x.H, down.KH, down.SH, down.PH);
            int odw = OUT_W(x.W, down.KW, down.SW, down.PW);
            id = empty_device_tensor(ctx, down.OC, odh, odw);
            run_conv_bn(q, K.k_convbn, x.buf,
                        down.dW, down.dB, down.dG, down.dBt, down.dMean, down.dVar, BN_EPS, id.buf,
                        down.IC, x.H, x.W, down.OC, down.KH, down.KW, odh, odw, down.SH, down.SW, down.PH, down.PW);
        } else {
            id = x; // 直接复用输入
        }

        // add + relu
        Tensor y = empty_device_tensor(ctx, y2.C, y2.H, y2.W);
        run_add(q, K.k_add, y2.buf, id.buf, y.buf, y.C, y.H, y.W);
        Tensor yr = empty_device_tensor(ctx, y.C, y.H, y.W);
        run_relu(q, K.k_relu, y.buf, yr.buf, y.C, y.H, y.W);
        return yr;
    }
};

static BasicBlock make_basic(int inC, int outC, int stride){
    BasicBlock b;
    // conv1: 3x3 s=stride p=1
    b.conv1.IC=inC;  b.conv1.OC=outC; b.conv1.KH=3; b.conv1.KW=3;
    b.conv1.SH=stride; b.conv1.SW=stride; b.conv1.PH=1; b.conv1.PW=1;
    // conv2: 3x3 s=1 p=1
    b.conv2.IC=outC; b.conv2.OC=outC; b.conv2.KH=3; b.conv2.KW=3;
    b.conv2.SH=1; b.conv2.SW=1; b.conv2.PH=1; b.conv2.PW=1;
    // downsample 1x1 s=stride when needed
    b.has_down = (stride!=1) || (inC!=outC);
    if (b.has_down){
        b.down.IC=inC; b.down.OC=outC; b.down.KH=1; b.down.KW=1;
        b.down.SH=stride; b.down.SW=stride; b.down.PH=0; b.down.PW=0;
    }
    return b;
}

int main(){
    srand((unsigned)time(NULL));

    // OpenCL
    cl_int err=CL_SUCCESS;
    cl_platform_id plat; cl_device_id dev;
    err = clGetPlatformIDs(1,&plat,NULL);
    err|= clGetDeviceIDs(plat, CL_DEVICE_TYPE_DEFAULT, 1,&dev,NULL);
    cl_context ctx = clCreateContext(NULL,1,&dev,NULL,NULL,&err);
    cl_command_queue q = clCreateCommandQueue(ctx,dev,0,&err);

    Kernels K; build_all_kernels(ctx, dev, K);

    // ===== 构建 ResNet18 =====
    // stem: conv7x7 s2 p3 -> bn -> relu -> maxpool3x3 s2 p1
    ConvBNParams stem; stem.IC=IN_C; stem.OC=64; stem.KH=7; stem.KW=7; stem.SH=2; stem.SW=2; stem.PH=3; stem.PW=3;
    init_convbn(stem);
    ConvBNParams d_stem; upload_convbn(ctx, stem, d_stem, &err);

    auto init_upload_block = [&](BasicBlock& b, BasicBlock& db){
        init_convbn(b.conv1); init_convbn(b.conv2);
        if (b.has_down) init_convbn(b.down);
        upload_convbn(ctx, b.conv1, db.conv1, &err);
        upload_convbn(ctx, b.conv2, db.conv2, &err);
        db.has_down = b.has_down;
        if (b.has_down) upload_convbn(ctx, b.down, db.down, &err);
    };

    // 4 stages: (64)x2, (128)x2, (256)x2, (512)x2 with strides 1,2,2,2
    BasicBlock st2_b1 = make_basic(64,  64, 1), st2_b2 = make_basic(64,  64, 1);
    BasicBlock st3_b1 = make_basic(64, 128, 2), st3_b2 = make_basic(128, 128,1);
    BasicBlock st4_b1 = make_basic(128,256, 2), st4_b2 = make_basic(256, 256,1);
    BasicBlock st5_b1 = make_basic(256,512, 2), st5_b2 = make_basic(512, 512,1);

    BasicBlock d_st2_b1, d_st2_b2, d_st3_b1, d_st3_b2, d_st4_b1, d_st4_b2, d_st5_b1, d_st5_b2;
    init_upload_block(st2_b1, d_st2_b1); init_upload_block(st2_b2, d_st2_b2);
    init_upload_block(st3_b1, d_st3_b1); init_upload_block(st3_b2, d_st3_b2);
    init_upload_block(st4_b1, d_st4_b1); init_upload_block(st4_b2, d_st4_b2);
    init_upload_block(st5_b1, d_st5_b1); init_upload_block(st5_b2, d_st5_b2);

    // FC
    const int NUM_CLASSES = 1000;
    std::vector<float> fcW((size_t)512*NUM_CLASSES), fcB(NUM_CLASSES);
    for (auto &v: fcW) v = frand()*0.1f;
    for (int i=0;i<NUM_CLASSES;++i) fcB[i]=frand()*0.1f;
    cl_mem d_fcW = make_device_buffer(ctx, fcW.data(), sizeof(float)*fcW.size(), &err);
    cl_mem d_fcB = make_device_buffer(ctx, fcB.data(), sizeof(float)*fcB.size(), &err);

    // 输入
    Tensor x0 = make_device_tensor(ctx, IN_C, IN_H, IN_W);

    // stem forward
    int oh = OUT_H(x0.H, stem.KH, stem.SH, stem.PH);
    int ow = OUT_W(x0.W, stem.KW, stem.SW, stem.PW);
    Tensor s1 = empty_device_tensor(ctx, stem.OC, oh, ow);
    run_conv_bn(q, K.k_convbn, x0.buf,
                d_stem.dW, d_stem.dB, d_stem.dG, d_stem.dBt, d_stem.dMean, d_stem.dVar, BN_EPS, s1.buf,
                stem.IC, x0.H, x0.W, stem.OC, stem.KH, stem.KW, oh, ow, stem.SH, stem.SW, stem.PH, stem.PW);
    Tensor s1r = empty_device_tensor(ctx, s1.C, s1.H, s1.W);
    run_relu(q, K.k_relu, s1.buf, s1r.buf, s1r.C, s1r.H, s1r.W);

    // maxpool 3x3 s2 p1
    int mph=3, mpw=3, mps=2, mpp=1;
    int poh = OUT_H(s1r.H, mph, mps, mpp);
    int pow = OUT_W(s1r.W, mpw, mps, mpp);
    Tensor p1 = empty_device_tensor(ctx, s1r.C, poh, pow);
    run_pool2d(q, K.k_pool, s1r.buf, p1.buf, p1.C, s1r.H, s1r.W, poh, pow, mph, mpw, mps, mps, mpp, mpp, 0, 0);

    // stages
    Tensor t = p1;
    t = d_st2_b1.forward(q, K, ctx, t);
    t = d_st2_b2.forward(q, K, ctx, t);
    t = d_st3_b1.forward(q, K, ctx, t);
    t = d_st3_b2.forward(q, K, ctx, t);
    t = d_st4_b1.forward(q, K, ctx, t);
    t = d_st4_b2.forward(q, K, ctx, t);
    t = d_st5_b1.forward(q, K, ctx, t);
    t = d_st5_b2.forward(q, K, ctx, t);

    // global avg pool -> 1x1
    Tensor g = empty_device_tensor(ctx, t.C, 1, 1);
    // run_pool2d(q, K.k_pool, t.buf, g.buf, g.C, t.H, t.W, 1, 1, t.H, t.W, 1, 1, 1, 0);
    // 全局平均池化: OH=OW=1, KH=IH, KW=IW, SH=SW=1, PH=PW=0, mode=1(avg), count_include_pad=0
    run_pool2d(q, K.k_pool, t.buf, g.buf,
            g.C,        // C
            t.H, t.W,   // IH, IW
            1, 1,       // OH, OW
            t.H, t.W,   // KH, KW
            1, 1,       // SH, SW
            0, 0,       // PH, PW   <-- 这两个是你漏掉的
            1, 0);      // mode=1(avg), count_include_pad=0

    // GEMM: A(1x512)=g, B(512xNUM_CLASSES)=fcW -> C(1xNUM_CLASSES)
    cl_mem d_logits = make_device_buffer(ctx, nullptr, sizeof(float)*NUM_CLASSES, &err);
    int M=1, Kdim=512, Ncls=NUM_CLASSES, transA=0, transB=0;
    run_gemm(q, K.k_gemm, g.buf, d_fcW, d_logits, M, Kdim, Ncls, transA, transB);

    // 读回 logits 并加偏置
    std::vector<float> logits(NUM_CLASSES, 0.0f);
    download_tensor(q, d_logits, logits.data(), sizeof(float)*NUM_CLASSES);
    for (int j=0;j<NUM_CLASSES;++j) logits[j]+=fcB[j];

    puts("\nResNet18 logits (first 10):");
    for (int i=0;i<10 && i<NUM_CLASSES;++i)
        printf("logit[%d] = %+9.6f\n", i, logits[i]);

    // 释放（示例简化）
    clReleaseMemObject(d_fcW); clReleaseMemObject(d_fcB); clReleaseMemObject(d_logits);
    clReleaseCommandQueue(q); clReleaseContext(ctx);
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
    return 0;
}
