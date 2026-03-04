#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cerrno>
#include <vector>
#include <string>
#include <cassert>
#include <iostream>
#include <limits>
#include <random>   // 固定随机种子
#include <fstream>
#include <sstream>
#include "../common/ventus_result_io.h"
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
// 这里采用不融合 BN 的版本
static const bool  USE_FUSED_CONV_BN = false;

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

// ======= 简单工具 =======
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

    // 如需更严谨的浮点可复现性，可改为：
    // const char* opts = "-cl-std=CL1.2 -cl-no-signed-zeros -cl-fp32-correctly-rounded-divide-sqrt";
    // err = clBuildProgram(prog, 1, &dev, opts, NULL, NULL);
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
static void zero_buffer(cl_command_queue q, cl_mem buf, size_t bytes){
    const cl_uint zero = 0;
    cl_int e = clEnqueueFillBuffer(q, buf, &zero, sizeof(zero), 0, bytes, 0, NULL, NULL);
    if(e!=CL_SUCCESS){ fprintf(stderr,"FillBuffer(0) failed: %d\n", e); exit(1); }
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
        K.k_conv      = clCreateKernel(K.prog_conv, "conv2d", &err);
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

// ======= 卷积参数 =======
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

struct RunOptions {
    bool check_cpu_ref = true;
    float atol = 1e-4f;
    float rtol = 1e-4f;
};

static void print_usage(const char* prog){
    fprintf(stderr,
        "Usage: %s [--cpu-ref|--no-cpu-ref] [--atol <float>] [--rtol <float>]\n"
        "  --cpu-ref      Enable CPU reference validation (default)\n"
        "  --no-cpu-ref   Disable CPU reference validation\n"
        "  --atol <v>     Absolute tolerance for validation (default: 1e-4)\n"
        "  --rtol <v>     Relative tolerance for validation (default: 1e-4)\n"
        "  --help         Show this help message\n",
        prog
    );
}

static bool parse_non_negative_float(const char* text, float& out){
    errno = 0;
    char* end = nullptr;
    const float v = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || v < 0.0f){
        return false;
    }
    out = v;
    return true;
}

static int parse_args(int argc, char** argv, RunOptions& opt){
    for (int i = 1; i < argc; ++i){
        const char* arg = argv[i];
        if (strcmp(arg, "--cpu-ref") == 0){
            opt.check_cpu_ref = true;
            continue;
        }
        if (strcmp(arg, "--no-cpu-ref") == 0){
            opt.check_cpu_ref = false;
            continue;
        }
        if (strcmp(arg, "--atol") == 0 || strcmp(arg, "--rtol") == 0){
            if (i + 1 >= argc){
                fprintf(stderr, "缺少参数值: %s\n", arg);
                print_usage(argv[0]);
                return 2;
            }
            float value = 0.0f;
            if (!parse_non_negative_float(argv[i + 1], value)){
                fprintf(stderr, "非法浮点参数: %s %s\n", arg, argv[i + 1]);
                return 2;
            }
            if (strcmp(arg, "--atol") == 0) opt.atol = value;
            else opt.rtol = value;
            ++i;
            continue;
        }
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0){
            print_usage(argv[0]);
            return 1;
        }
        fprintf(stderr, "未知参数: %s\n", arg);
        print_usage(argv[0]);
        return 2;
    }
    return 0;
}

struct HostTensor3D {
    int C = 0;
    int H = 0;
    int W = 0;
    std::vector<float> data;
    HostTensor3D() = default;
    HostTensor3D(int c, int h, int w) : C(c), H(h), W(w), data((size_t)c * h * w) {}
};

static inline size_t idx3d(int c, int y, int x, int H, int W){
    return (size_t)c * (size_t)H * (size_t)W + (size_t)y * (size_t)W + (size_t)x;
}

static HostTensor3D cpu_conv2d(const HostTensor3D& in, const ConvParams& p, int do_relu){
    HostTensor3D out(p.OC, OUT_H(in.H, p.KH, p.SH, p.PH), OUT_W(in.W, p.KW, p.SW, p.PW));
    for (int oc = 0; oc < out.C; ++oc){
        for (int oy = 0; oy < out.H; ++oy){
            for (int ox = 0; ox < out.W; ++ox){
                float sum = p.B.empty() ? 0.0f : p.B[oc];
                for (int ic = 0; ic < p.IC; ++ic){
                    for (int ky = 0; ky < p.KH; ++ky){
                        const int iy = oy * p.SH - p.PH + ky;
                        if (iy < 0 || iy >= in.H) continue;
                        for (int kx = 0; kx < p.KW; ++kx){
                            const int ix = ox * p.SW - p.PW + kx;
                            if (ix < 0 || ix >= in.W) continue;
                            const size_t in_idx = idx3d(ic, iy, ix, in.H, in.W);
                            const size_t w_idx = (size_t)oc * p.IC * p.KH * p.KW
                                               + (size_t)ic * p.KH * p.KW
                                               + (size_t)ky * p.KW + kx;
                            sum += in.data[in_idx] * p.W[w_idx];
                        }
                    }
                }
                if (do_relu && sum < 0.0f) sum = 0.0f;
                out.data[idx3d(oc, oy, ox, out.H, out.W)] = sum;
            }
        }
    }
    return out;
}

static HostTensor3D cpu_relu3d(const HostTensor3D& in){
    HostTensor3D out(in.C, in.H, in.W);
    for (size_t i = 0; i < in.data.size(); ++i){
        const float v = in.data[i];
        out.data[i] = (v > 0.0f) ? v : 0.0f;
    }
    return out;
}

static HostTensor3D cpu_pool2d(const HostTensor3D& in,
                               int KH, int KW, int SH, int SW, int PH, int PW,
                               int mode, int count_include_pad){
    HostTensor3D out(in.C, OUT_H(in.H, KH, SH, PH), OUT_W(in.W, KW, SW, PW));
    for (int c = 0; c < out.C; ++c){
        for (int oy = 0; oy < out.H; ++oy){
            for (int ox = 0; ox < out.W; ++ox){
                float acc = (mode == 0) ? -std::numeric_limits<float>::infinity() : 0.0f;
                int count = 0;
                for (int ky = 0; ky < KH; ++ky){
                    const int iy = oy * SH - PH + ky;
                    for (int kx = 0; kx < KW; ++kx){
                        const int ix = ox * SW - PW + kx;
                        const bool valid = (iy >= 0 && iy < in.H && ix >= 0 && ix < in.W);
                        if (mode == 0){
                            if (valid){
                                const float v = in.data[idx3d(c, iy, ix, in.H, in.W)];
                                acc = fmax(acc, v);
                            }
                            continue;
                        }
                        const float v = valid ? in.data[idx3d(c, iy, ix, in.H, in.W)] : 0.0f;
                        if (count_include_pad || valid){
                            acc += v;
                            ++count;
                        }
                    }
                }
                if (mode == 1) acc = (count > 0) ? (acc / (float)count) : 0.0f;
                out.data[idx3d(c, oy, ox, out.H, out.W)] = acc;
            }
        }
    }
    return out;
}

static std::vector<float> cpu_gemm(const std::vector<float>& A,
                                   const std::vector<float>& B,
                                   int M, int K, int N,
                                   int do_trans_a, int do_trans_b){
    std::vector<float> C((size_t)M * N, 0.0f);
    for (int row = 0; row < M; ++row){
        for (int col = 0; col < N; ++col){
            float sum = 0.0f;
            for (int k = 0; k < K; ++k){
                const float a_val = do_trans_a ? A[(size_t)k * M + row] : A[(size_t)row * K + k];
                const float b_val = do_trans_b ? B[(size_t)col * K + k] : B[(size_t)k * N + col];
                sum += a_val * b_val;
            }
            C[(size_t)row * N + col] = sum;
        }
    }
    return C;
}

static std::vector<float> cpu_add_1d(const std::vector<float>& a, const std::vector<float>& b){
    if (a.size() != b.size()){
        fprintf(stderr, "cpu_add_1d size mismatch: %zu vs %zu\n", a.size(), b.size());
        exit(1);
    }
    std::vector<float> out(a.size(), 0.0f);
    for (size_t i = 0; i < a.size(); ++i){
        out[i] = a[i] + b[i];
    }
    return out;
}

static std::vector<float> cpu_relu_1d(const std::vector<float>& in){
    std::vector<float> out(in.size(), 0.0f);
    for (size_t i = 0; i < in.size(); ++i){
        const float v = in[i];
        out[i] = (v > 0.0f) ? v : 0.0f;
    }
    return out;
}

struct CpuRefContext {
    const Tensor& x0;
    const ConvParams& conv1;
    const ConvParams& conv2;
    const ConvParams& conv3;
    const ConvParams& conv4;
    const ConvParams& conv5;
    const std::vector<float>& fcW1;
    const std::vector<float>& fcB1;
    const std::vector<float>& fcW2;
    const std::vector<float>& fcB2;
    const std::vector<float>& fcW3;
    const std::vector<float>& fcB3;
    int fc1_size;
    int fc2_size;

    CpuRefContext(const Tensor& x0_,
                  const ConvParams& conv1_,
                  const ConvParams& conv2_,
                  const ConvParams& conv3_,
                  const ConvParams& conv4_,
                  const ConvParams& conv5_,
                  const std::vector<float>& fcW1_,
                  const std::vector<float>& fcB1_,
                  const std::vector<float>& fcW2_,
                  const std::vector<float>& fcB2_,
                  const std::vector<float>& fcW3_,
                  const std::vector<float>& fcB3_,
                  int fc1_size_,
                  int fc2_size_)
        : x0(x0_),
          conv1(conv1_),
          conv2(conv2_),
          conv3(conv3_),
          conv4(conv4_),
          conv5(conv5_),
          fcW1(fcW1_),
          fcB1(fcB1_),
          fcW2(fcW2_),
          fcB2(fcB2_),
          fcW3(fcW3_),
          fcB3(fcB3_),
          fc1_size(fc1_size_),
          fc2_size(fc2_size_) {}
};

static void validate_fc_shapes(const CpuRefContext& ctx, int flat_k){
    const size_t w1_expect = (size_t)ctx.fc1_size * flat_k;
    const size_t w2_expect = (size_t)ctx.fc2_size * ctx.fc1_size;
    const size_t w3_expect = (size_t)NUM_CLASSES * ctx.fc2_size;
    if (ctx.fcW1.size() != w1_expect || ctx.fcB1.size() != (size_t)ctx.fc1_size ||
        ctx.fcW2.size() != w2_expect || ctx.fcB2.size() != (size_t)ctx.fc2_size ||
        ctx.fcW3.size() != w3_expect || ctx.fcB3.size() != (size_t)NUM_CLASSES){
        fprintf(stderr, "CPU reference FC 参数尺寸不匹配\n");
        exit(1);
    }
}

static std::vector<float> run_cpu_reference_logits(const CpuRefContext& ctx){
    HostTensor3D x(ctx.x0.C, ctx.x0.H, ctx.x0.W);
    x.data = ctx.x0.host;
    const int P1_K = 3, P1_S = 2, P_PAD = 0;

    HostTensor3D y1 = cpu_conv2d(x, ctx.conv1, 0);
    HostTensor3D r1 = cpu_relu3d(y1);
    HostTensor3D p1 = cpu_pool2d(r1, P1_K, P1_K, P1_S, P1_S, P_PAD, P_PAD, 0, 0);

    HostTensor3D y2 = cpu_conv2d(p1, ctx.conv2, 0);
    HostTensor3D r2 = cpu_relu3d(y2);
    HostTensor3D p2 = cpu_pool2d(r2, P1_K, P1_K, P1_S, P1_S, P_PAD, P_PAD, 0, 0);

    HostTensor3D y3 = cpu_conv2d(p2, ctx.conv3, 0);
    HostTensor3D r3 = cpu_relu3d(y3);

    HostTensor3D y4 = cpu_conv2d(r3, ctx.conv4, 0);
    HostTensor3D r4 = cpu_relu3d(y4);

    HostTensor3D y5 = cpu_conv2d(r4, ctx.conv5, 0);
    HostTensor3D r5 = cpu_relu3d(y5);
    HostTensor3D p5 = cpu_pool2d(r5, P1_K, P1_K, P1_S, P1_S, P_PAD, P_PAD, 0, 0);

    const int flat_k = p5.C * p5.H * p5.W;
    validate_fc_shapes(ctx, flat_k);

    const std::vector<float> fc1_mm = cpu_gemm(p5.data, ctx.fcW1, 1, flat_k, ctx.fc1_size, 0, 0);
    const std::vector<float> fc1_out = cpu_add_1d(fc1_mm, ctx.fcB1);
    const std::vector<float> fc1_relu = cpu_relu_1d(fc1_out);

    const std::vector<float> fc2_mm = cpu_gemm(fc1_relu, ctx.fcW2, 1, ctx.fc1_size, ctx.fc2_size, 0, 0);
    const std::vector<float> fc2_out = cpu_add_1d(fc2_mm, ctx.fcB2);
    const std::vector<float> fc2_relu = cpu_relu_1d(fc2_out);

    const std::vector<float> fc3_mm = cpu_gemm(fc2_relu, ctx.fcW3, 1, ctx.fc2_size, NUM_CLASSES, 0, 0);
    return cpu_add_1d(fc3_mm, ctx.fcB3);
}

static bool check_cpu_reference(const std::vector<float>& got,
                                const std::vector<float>& ref,
                                float atol,
                                float rtol){
    if (got.size() != ref.size()){
        fprintf(stderr, "[CPU-REF] size mismatch: got=%zu ref=%zu\n", got.size(), ref.size());
        return false;
    }
    bool ok = true;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (size_t i = 0; i < got.size(); ++i){
        const float abs_err = fabs(got[i] - ref[i]);
        const float rel_err = abs_err / (fabs(ref[i]) + 1e-20f);
        const float tol = atol + rtol * fabs(ref[i]);
        if (abs_err > max_abs) max_abs = abs_err;
        if (rel_err > max_rel) max_rel = rel_err;
        if (abs_err > tol){
            fprintf(stderr,
                "[CPU-REF] mismatch idx=%zu got=%.8f ref=%.8f abs=%.8g tol=%.8g\n",
                i, got[i], ref[i], abs_err, tol);
            ok = false;
        }
    }
    if (ok){
        printf("[CPU-REF] PASS atol=%.3e rtol=%.3e max_abs=%.6g max_rel=%.6g\n",
               atol, rtol, max_abs, max_rel);
    } else {
        fprintf(stderr,
            "[CPU-REF] FAIL atol=%.3e rtol=%.3e max_abs=%.6g max_rel=%.6g\n",
            atol, rtol, max_abs, max_rel);
    }
    return ok;
}

// ======= 运行封装 =======
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

int main(int argc, char** argv) {
    RunOptions run_options;
    const int parse_rc = parse_args(argc, argv, run_options);
    if (parse_rc == 1) return 0;
    if (parse_rc != 0) return parse_rc;

    // 全局基种子（可改以得到另一组可复现结果）
    const uint32_t SEED_BASE = 0x13572468u;

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

    // ===== 输入（固定种子）=====
    Tensor x0 = Tensor(IN_C, IN_H, IN_W);
    {
        std::vector<float>& v = x0.host;
        fill_uniform(v, -1.f, 1.f, seed_mix(SEED_BASE, 1));
    }
    x0.buf = make_device_buffer(ctx, x0.host.data(), x0.bytes(), &err);

    // ===== 构建 AlexNet（固定种子）=====
    // conv1: 11x11, s=4, p=2
    ConvParams conv1; conv1.IC = IN_C; conv1.OC = 64;  conv1.KH = 11; conv1.KW = 11; conv1.SH = 4; conv1.SW = 4; conv1.PH = 2; conv1.PW = 2;
    init_conv_seeded(conv1, seed_mix(SEED_BASE, 101), seed_mix(SEED_BASE, 102));
    // conv2: 5x5, s=1, p=2
    ConvParams conv2; conv2.IC = 64;  conv2.OC = 192; conv2.KH = 5;  conv2.KW = 5;  conv2.SH = 1; conv2.SW = 1; conv2.PH = 2; conv2.PW = 2;
    init_conv_seeded(conv2, seed_mix(SEED_BASE, 201), seed_mix(SEED_BASE, 202));
    // conv3: 3x3, s=1, p=1
    ConvParams conv3; conv3.IC = 192; conv3.OC = 384; conv3.KH = 3;  conv3.KW = 3;  conv3.SH = 1; conv3.SW = 1; conv3.PH = 1; conv3.PW = 1;
    init_conv_seeded(conv3, seed_mix(SEED_BASE, 301), seed_mix(SEED_BASE, 302));
    // conv4: 3x3, s=1, p=1
    ConvParams conv4; conv4.IC = 384; conv4.OC = 256; conv4.KH = 3;  conv4.KW = 3;  conv4.SH = 1; conv4.SW = 1; conv4.PH = 1; conv4.PW = 1;
    init_conv_seeded(conv4, seed_mix(SEED_BASE, 401), seed_mix(SEED_BASE, 402));
    // conv5: 3x3, s=1, p=1
    ConvParams conv5; conv5.IC = 256; conv5.OC = 256; conv5.KH = 3;  conv5.KW = 3;  conv5.SH = 1; conv5.SW = 1; conv5.PH = 1; conv5.PW = 1;
    init_conv_seeded(conv5, seed_mix(SEED_BASE, 501), seed_mix(SEED_BASE, 502));

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

    // ======== Forward: conv1 -> relu -> pool ========
    Tensor y1( conv1.OC, OUT_H(x0.H, conv1.KH, conv1.SH, conv1.PH), OUT_W(x0.W, conv1.KW, conv1.SW, conv1.PW) );
    y1.buf = make_device_buffer(ctx, nullptr, y1.bytes(), &err);
    zero_buffer(q, y1.buf, y1.bytes());
    run_conv_layer(q, K.k_conv, x0.buf, conv1.dW, conv1.dB, y1.buf,
                   conv1.IC, x0.H, x0.W, conv1.OC, conv1.KH, conv1.KW, y1.H, y1.W,
                   conv1.SH, conv1.SW, conv1.PH, conv1.PW, 0);

    Tensor r1 = y1; r1.buf = make_device_buffer(ctx, nullptr, r1.bytes(), &err);
    zero_buffer(q, r1.buf, r1.bytes());
    run_relu3d(q, K.k_relu, y1.buf, r1.buf, r1.C, r1.H, r1.W);

    // pool1: 3x3 s=2 p=0
    const int P1_K=3, P1_S=2, P_PAD=0;
    Tensor p1( r1.C, OUT_H(r1.H, P1_K, P1_S, P_PAD), OUT_W(r1.W, P1_K, P1_S, P_PAD) );
    p1.buf = make_device_buffer(ctx, nullptr, p1.bytes(), &err);
    zero_buffer(q, p1.buf, p1.bytes());
    run_pool2d(q, K.k_pool, r1.buf, p1.buf, p1.C, r1.H, r1.W, p1.H, p1.W,
               P1_K, P1_K, P1_S, P1_S, P_PAD, P_PAD, 0, 0);

    // ======== conv2 -> relu -> pool ========
    Tensor y2( conv2.OC, OUT_H(p1.H, conv2.KH, conv2.SH, conv2.PH), OUT_W(p1.W, conv2.KW, conv2.SW, conv2.PW) );
    y2.buf = make_device_buffer(ctx, nullptr, y2.bytes(), &err);
    zero_buffer(q, y2.buf, y2.bytes());
    run_conv_layer(q, K.k_conv, p1.buf, conv2.dW, conv2.dB, y2.buf,
                   conv2.IC, p1.H, p1.W, conv2.OC, conv2.KH, conv2.KW, y2.H, y2.W,
                   conv2.SH, conv2.SW, conv2.PH, conv2.PW, 0);

    Tensor r2 = y2; r2.buf = make_device_buffer(ctx, nullptr, r2.bytes(), &err);
    zero_buffer(q, r2.buf, r2.bytes());
    run_relu3d(q, K.k_relu, y2.buf, r2.buf, r2.C, r2.H, r2.W);

    Tensor p2( r2.C, OUT_H(r2.H, P1_K, P1_S, P_PAD), OUT_W(r2.W, P1_K, P1_S, P_PAD) );
    p2.buf = make_device_buffer(ctx, nullptr, p2.bytes(), &err);
    zero_buffer(q, p2.buf, p2.bytes());
    run_pool2d(q, K.k_pool, r2.buf, p2.buf, p2.C, r2.H, r2.W, p2.H, p2.W,
               P1_K, P1_K, P1_S, P1_S, P_PAD, P_PAD, 0, 0);

    // ======== conv3 -> relu ========
    Tensor y3( conv3.OC, OUT_H(p2.H, conv3.KH, conv3.SH, conv3.PH), OUT_W(p2.W, conv3.KW, conv3.SW, conv3.PW) );
    y3.buf = make_device_buffer(ctx, nullptr, y3.bytes(), &err);
    zero_buffer(q, y3.buf, y3.bytes());
    run_conv_layer(q, K.k_conv, p2.buf, conv3.dW, conv3.dB, y3.buf,
                   conv3.IC, p2.H, p2.W, conv3.OC, conv3.KH, conv3.KW, y3.H, y3.W,
                   conv3.SH, conv3.SW, conv3.PH, conv3.PW, 0);

    Tensor r3 = y3; r3.buf = make_device_buffer(ctx, nullptr, r3.bytes(), &err);
    zero_buffer(q, r3.buf, r3.bytes());
    run_relu3d(q, K.k_relu, y3.buf, r3.buf, r3.C, r3.H, r3.W);

    // ======== conv4 -> relu ========
    Tensor y4( conv4.OC, OUT_H(r3.H, conv4.KH, conv4.SH, conv4.PH), OUT_W(r3.W, conv4.KW, conv4.SW, conv4.PW) );
    y4.buf = make_device_buffer(ctx, nullptr, y4.bytes(), &err);
    zero_buffer(q, y4.buf, y4.bytes());
    run_conv_layer(q, K.k_conv, r3.buf, conv4.dW, conv4.dB, y4.buf,
                   conv4.IC, r3.H, r3.W, conv4.OC, conv4.KH, conv4.KW, y4.H, y4.W,
                   conv4.SH, conv4.SW, conv4.PH, conv4.PW, 0);

    Tensor r4 = y4; r4.buf = make_device_buffer(ctx, nullptr, r4.bytes(), &err);
    zero_buffer(q, r4.buf, r4.bytes());
    run_relu3d(q, K.k_relu, y4.buf, r4.buf, r4.C, r4.H, r4.W);

    // ======== conv5 -> relu -> pool ========
    Tensor y5( conv5.OC, OUT_H(r4.H, conv5.KH, conv5.SH, conv5.PH), OUT_W(r4.W, conv5.KW, conv5.SW, conv5.PW) );
    y5.buf = make_device_buffer(ctx, nullptr, y5.bytes(), &err);
    zero_buffer(q, y5.buf, y5.bytes());
    run_conv_layer(q, K.k_conv, r4.buf, conv5.dW, conv5.dB, y5.buf,
                   conv5.IC, r4.H, r4.W, conv5.OC, conv5.KH, conv5.KW, y5.H, y5.W,
                   conv5.SH, conv5.SW, conv5.PH, conv5.PW, 0);

    Tensor r5 = y5; r5.buf = make_device_buffer(ctx, nullptr, r5.bytes(), &err);
    zero_buffer(q, r5.buf, r5.bytes());
    run_relu3d(q, K.k_relu, y5.buf, r5.buf, r5.C, r5.H, r5.W);

    // pool5: 3x3 s=2 p=0
    Tensor p5( r5.C, OUT_H(r5.H, P1_K, P1_S, P_PAD), OUT_W(r5.W, P1_K, P1_S, P_PAD) );
    p5.buf = make_device_buffer(ctx, nullptr, p5.bytes(), &err);
    zero_buffer(q, p5.buf, p5.bytes());
    run_pool2d(q, K.k_pool, r5.buf, p5.buf, p5.C, r5.H, r5.W, p5.H, p5.W,
               P1_K, P1_K, P1_S, P1_S, P_PAD, P_PAD, 0, 0);

    // ======== Flatten + FC1 -> ReLU -> FC2 -> ReLU -> FC3 ========
    const int FC1_SIZE = 4096;
    const int FC2_SIZE = 4096;
    const int FLAT_K   = p5.C * p5.H * p5.W;   // 不写死 256*6*6，自动匹配实际尺寸

    // 初始化 FC 权重（固定种子，范围[-0.1,0.1]）
    std::vector<float> fcW1(FC1_SIZE * FLAT_K), fcB1(FC1_SIZE);
    std::vector<float> fcW2(FC2_SIZE * FC1_SIZE), fcB2(FC2_SIZE);
    std::vector<float> fcW3(NUM_CLASSES * FC2_SIZE), fcB3(NUM_CLASSES);
    fill_uniform(fcW1, -0.1f, 0.1f, seed_mix(SEED_BASE, 601));
    fill_uniform(fcB1, -0.1f, 0.1f, seed_mix(SEED_BASE, 602));
    fill_uniform(fcW2, -0.1f, 0.1f, seed_mix(SEED_BASE, 701));
    fill_uniform(fcB2, -0.1f, 0.1f, seed_mix(SEED_BASE, 702));
    fill_uniform(fcW3, -0.1f, 0.1f, seed_mix(SEED_BASE, 801));
    fill_uniform(fcB3, -0.1f, 0.1f, seed_mix(SEED_BASE, 802));

    cl_mem d_fcW1 = make_device_buffer(ctx, fcW1.data(), sizeof(float) * fcW1.size(), &err);
    cl_mem d_fcB1 = make_device_buffer(ctx, fcB1.data(), sizeof(float) * fcB1.size(), &err);
    cl_mem d_fcW2 = make_device_buffer(ctx, fcW2.data(), sizeof(float) * fcW2.size(), &err);
    cl_mem d_fcB2 = make_device_buffer(ctx, fcB2.data(), sizeof(float) * fcB2.size(), &err);
    cl_mem d_fcW3 = make_device_buffer(ctx, fcW3.data(), sizeof(float) * fcW3.size(), &err);
    cl_mem d_fcB3 = make_device_buffer(ctx, fcB3.data(), sizeof(float) * fcB3.size(), &err);

    // FC1
    Tensor fc1_mm(1,1,FC1_SIZE); fc1_mm.buf = make_device_buffer(ctx,nullptr,sizeof(float)*FC1_SIZE,&err);
    zero_buffer(q, fc1_mm.buf, sizeof(float)*FC1_SIZE);
    run_gemm(q, K.k_gemm, p5.buf, d_fcW1, fc1_mm.buf, 1, FLAT_K, FC1_SIZE, 0, 0);

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

    // 下载并输出最后一层结果
    std::vector<float> out(NUM_CLASSES);
    download_tensor(q, logits.buf, out.data(), sizeof(float)*NUM_CLASSES);

    printf("== AlexNet 最后一层 logits (deterministic) ==\n");
    for (int i=0;i<NUM_CLASSES;++i){
        printf("%d %.6f\n", i, out[i]);
    }
    ventus_write_final_hex(out);

    int exit_code = 0;
    if (run_options.check_cpu_ref){
        const CpuRefContext ref_ctx{
            x0, conv1, conv2, conv3, conv4, conv5,
            fcW1, fcB1, fcW2, fcB2, fcW3, fcB3,
            FC1_SIZE, FC2_SIZE
        };
        const std::vector<float> ref = run_cpu_reference_logits(ref_ctx);
        if (!check_cpu_reference(out, ref, run_options.atol, run_options.rtol)){
            exit_code = 3;
        }
    } else {
        printf("[CPU-REF] SKIP\n");
    }

    // 资源释放
    clReleaseMemObject(x0.buf);
    clReleaseMemObject(y1.buf); clReleaseMemObject(r1.buf); clReleaseMemObject(p1.buf);
    clReleaseMemObject(y2.buf); clReleaseMemObject(r2.buf); clReleaseMemObject(p2.buf);
    clReleaseMemObject(y3.buf); clReleaseMemObject(r3.buf);
    clReleaseMemObject(y4.buf); clReleaseMemObject(r4.buf);
    clReleaseMemObject(y5.buf); clReleaseMemObject(r5.buf); clReleaseMemObject(p5.buf);
    clReleaseMemObject(fc1_mm.buf); clReleaseMemObject(fc1_out.buf); clReleaseMemObject(fc1_relu.buf);
    clReleaseMemObject(fc2_mm.buf); clReleaseMemObject(fc2_out.buf); clReleaseMemObject(fc2_relu.buf);
    clReleaseMemObject(fc3_mm.buf); clReleaseMemObject(logits.buf);

    clReleaseMemObject(conv1.dW); clReleaseMemObject(conv1.dB);
    clReleaseMemObject(conv2.dW); clReleaseMemObject(conv2.dB);
    clReleaseMemObject(conv3.dW); clReleaseMemObject(conv3.dB);
    clReleaseMemObject(conv4.dW); clReleaseMemObject(conv4.dB);
    clReleaseMemObject(conv5.dW); clReleaseMemObject(conv5.dB);

    clReleaseMemObject(d_fcW1); clReleaseMemObject(d_fcB1);
    clReleaseMemObject(d_fcW2); clReleaseMemObject(d_fcB2);
    clReleaseMemObject(d_fcW3); clReleaseMemObject(d_fcB3);

    if (K.k_conv)   clReleaseKernel(K.k_conv);
    if (K.k_convbn) clReleaseKernel(K.k_convbn);
    if (K.k_bn)     clReleaseKernel(K.k_bn);
    if (K.k_relu)   clReleaseKernel(K.k_relu);
    if (K.k_add)    clReleaseKernel(K.k_add);
    if (K.k_pool)   clReleaseKernel(K.k_pool);
    if (K.k_gemm)   clReleaseKernel(K.k_gemm);

    if (K.prog_conv)    clReleaseProgram(K.prog_conv);
    if (K.prog_convbn)  clReleaseProgram(K.prog_convbn);
    if (K.prog_bn)      clReleaseProgram(K.prog_bn);
    if (K.prog_relu)    clReleaseProgram(K.prog_relu);
    if (K.prog_add)     clReleaseProgram(K.prog_add);
    if (K.prog_pool)    clReleaseProgram(K.prog_pool);
    if (K.prog_gemm)    clReleaseProgram(K.prog_gemm);

    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return exit_code;
}
