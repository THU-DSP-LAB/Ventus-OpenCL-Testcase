// main.cc — host code with per-layer check & block count (random data; kernel from file)
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

/* ── 默认参数（可改） ───────────────── */
#define DEF_IN_C     3
#define DEF_IN_H     16
#define DEF_IN_W     16
#define DEF_OUT_C    8
#define DEF_KH       3
#define DEF_KW       3
#define DEF_STRIDE_H 1
#define DEF_STRIDE_W 1
#define DEF_PAD_H    1
#define DEF_PAD_W    1
#define DEF_DO_RELU  1
#define KERNEL_PATH "conv.cl"

/* ── 小工具 ─────────────────────────── */
static inline float frand(){ return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }

static char* load_kernel_source(const char* file){
    FILE* fp = fopen(file, "rb");
    if(!fp){ fprintf(stderr, "无法打开 kernel 文件: %s\n", file); exit(1); }
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); rewind(fp);
    char* src = (char*)malloc(sz + 1);
    fread(src, 1, sz, fp); src[sz] = '\0'; fclose(fp);
    return src;
}

/* CPU reference：与 kernel 等价 */
static void conv2d_reference(
    const float* in, const float* w, const float* b, float* out,
    int in_c, int in_h, int in_w,
    int out_c, int k_h, int k_w,
    int out_h, int out_w,
    int s_h, int s_w, int p_h, int p_w, int do_relu)
{
    const int inHW  = in_h * in_w;
    const int outHW = out_h * out_w;
    for(int oc=0; oc<out_c; ++oc){
        for(int oy=0; oy<out_h; ++oy){
            for(int ox=0; ox<out_w; ++ox){
                float sum = b ? b[oc] : 0.0f;
                for(int ic=0; ic<in_c; ++ic){
                    for(int ky=0; ky<k_h; ++ky){
                        int iy = oy*s_h - p_h + ky;
                        if(iy<0 || iy>=in_h) continue;
                        for(int kx=0; kx<k_w; ++kx){
                            int ix = ox*s_w - p_w + kx;
                            if(ix<0 || ix>=in_w) continue;
                            int in_idx = ic*inHW + iy*in_w + ix;
                            int w_idx  = oc*(in_c*k_h*k_w) + ic*(k_h*k_w) + ky*k_w + kx;
                            sum += in[in_idx] * w[w_idx];
                        }
                    }
                }
                if (do_relu && sum < 0.0f) sum = 0.0f;
                out[oc*outHW + oy*out_w + ox] = sum;
            }
        }
    }
}

/* 误差打印与统计（返回不合格比例%） */
static double compare_and_report(const char* tag,
                                 const float* cl, const float* ref, int n,
                                 double rtol=1e-4, double atol=1e-5)
{
    int bad=0, preview = n<10? n:10;
    printf("\n--- %s (%d elements) ---\n", tag, n);
    printf("idx\tOpenCL\t\tRef\t\tAbsErr\tRelErr\n");
    for(int i=0;i<n;++i){
        double absErr=fabs((double)cl[i]-(double)ref[i]);
        double relErr=(fabs(ref[i])>1e-12)? absErr/fabs(ref[i]) : 0.0;
        if(i<preview){
            printf("%4d\t%+9.6f\t%+9.6f\t%1.3e\t%1.3e\n",
                   i, cl[i], ref[i], absErr, relErr);
        }
        if(!(absErr<=atol || relErr<=rtol)) ++bad;
    }
    if(n>preview) puts("...(省略)");
    double ratio = 100.0 * (double)bad / (double)n;
    printf("不符合阈值元素: %d / %d (%.3f%%)\n", bad, n, ratio);
    return ratio;
}

/* 简单命令行参数解析：--in_c 3 --in_h 28 ... */
static void parse_args(int argc, char** argv,
    int &in_c, int &in_h, int &in_w, int &out_c, int &k_h, int &k_w,
    int &s_h, int &s_w, int &p_h, int &p_w, int &do_relu)
{
    for(int i=1;i<argc;++i){
        const char* a = argv[i];
        auto next = [&](int &dst){ if(i+1<argc) dst = atoi(argv[++i]); };
        if(!strcmp(a,"--in_c")) next(in_c);
        else if(!strcmp(a,"--in_h")) next(in_h);
        else if(!strcmp(a,"--in_w")) next(in_w);
        else if(!strcmp(a,"--out_c")) next(out_c);
        else if(!strcmp(a,"--kh")) next(k_h);
        else if(!strcmp(a,"--kw")) next(k_w);
        else if(!strcmp(a,"--sh")) next(s_h);
        else if(!strcmp(a,"--sw")) next(s_w);
        else if(!strcmp(a,"--ph")) next(p_h);
        else if(!strcmp(a,"--pw")) next(p_w);
        else if(!strcmp(a,"--relu")) next(do_relu); // 1/0
        else if(!strcmp(a,"-h")||!strcmp(a,"--help")){
            printf("Usage: %s [--in_c C] [--in_h H] [--in_w W] [--out_c OC]\n"
                   "              [--kh KH] [--kw KW] [--sh SH] [--sw SW]\n"
                   "              [--ph PH] [--pw PW] [--relu 1|0]\n", argv[0]);
            exit(0);
        }
    }
}

/* ── main ──────────────────────────── */
int main(int argc, char** argv){
    srand((unsigned)time(NULL));

    // 1) 形状参数（可被命令行覆盖）
    int in_c = DEF_IN_C, in_h = DEF_IN_H, in_w = DEF_IN_W;
    int out_c = DEF_OUT_C, k_h = DEF_KH, k_w = DEF_KW;
    int s_h = DEF_STRIDE_H, s_w = DEF_STRIDE_W;
    int p_h = DEF_PAD_H,    p_w = DEF_PAD_W;
    int do_relu = DEF_DO_RELU;
    parse_args(argc, argv, in_c, in_h, in_w, out_c, k_h, k_w, s_h, s_w, p_h, p_w, do_relu);

    int out_h = (in_h + 2*p_h - k_h)/s_h + 1;
    int out_w = (in_w + 2*p_w - k_w)/s_w + 1;
    if(out_h<=0 || out_w<=0){
        fprintf(stderr, "非法输出尺寸: out_h=%d out_w=%d（检查 KH/KW/stride/pad）\n", out_h, out_w);
        return 1;
    }

    const int in_sz   = in_c * in_h * in_w;
    const int w_sz    = out_c * in_c * k_h * k_w;
    const int bias_sz = out_c;
    const int out_sz  = out_c * out_h * out_w;

    // 2) 随机数据
    float *input  = (float*)malloc(sizeof(float)*in_sz);
    float *weight = (float*)malloc(sizeof(float)*w_sz);
    float *bias   = (float*)malloc(sizeof(float)*bias_sz);
    float *out_cl = (float*)malloc(sizeof(float)*out_sz);
    float *out_ref= (float*)malloc(sizeof(float)*out_sz);
    for(int i=0;i<in_sz;++i)   input[i]  = frand();
    for(int i=0;i<w_sz;++i)    weight[i] = frand();
    for(int i=0;i<bias_sz;++i) bias[i]   = frand();

    // 3) OpenCL 平台/设备/上下文/队列
    cl_int err = CL_SUCCESS;
    cl_platform_id plat; cl_device_id dev;
    err = clGetPlatformIDs(1, &plat, NULL);
    err|= clGetDeviceIDs(plat, CL_DEVICE_TYPE_DEFAULT, 1, &dev, NULL);
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err); // OpenCL 1.2 简洁用法

    // 4) program / kernel
    char* src = load_kernel_source(KERNEL_PATH);
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, NULL, &err);
    free(src);
    err = clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
    if(err != CL_SUCCESS){
        size_t log_sz=0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        char* log=(char*)malloc(log_sz+1);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL); log[log_sz]='\0';
        fprintf(stderr, "Build log:\n%s\n", log); free(log);
        return 1;
    }
    cl_kernel k = clCreateKernel(prog, "conv2d", &err);

    // 5) 缓冲区
    cl_mem buf_in  = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, sizeof(float)*in_sz,   input,  &err);
    cl_mem buf_w   = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, sizeof(float)*w_sz,    weight, &err);
    cl_mem buf_b   = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, sizeof(float)*bias_sz, bias,   &err);
    cl_mem buf_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,                              sizeof(float)*out_sz, NULL,   &err);

    // 6) 设置 kernel 参数（全部传变量地址）
    err  = clSetKernelArg(k, 0,  sizeof(cl_mem), &buf_in);
    err |= clSetKernelArg(k, 1,  sizeof(cl_mem), &buf_w);
    err |= clSetKernelArg(k, 2,  sizeof(cl_mem), &buf_b);
    err |= clSetKernelArg(k, 3,  sizeof(cl_mem), &buf_out);
    err |= clSetKernelArg(k, 4,  sizeof(int),    &in_c);
    err |= clSetKernelArg(k, 5,  sizeof(int),    &in_h);
    err |= clSetKernelArg(k, 6,  sizeof(int),    &in_w);
    err |= clSetKernelArg(k, 7,  sizeof(int),    &out_c);
    err |= clSetKernelArg(k, 8,  sizeof(int),    &k_h);
    err |= clSetKernelArg(k, 9,  sizeof(int),    &k_w);
    err |= clSetKernelArg(k,10,  sizeof(int),    &out_h);
    err |= clSetKernelArg(k,11,  sizeof(int),    &out_w);
    err |= clSetKernelArg(k,12,  sizeof(int),    &s_h);
    err |= clSetKernelArg(k,13,  sizeof(int),    &s_w);
    err |= clSetKernelArg(k,14,  sizeof(int),    &p_h);
    err |= clSetKernelArg(k,15,  sizeof(int),    &p_w);
    err |= clSetKernelArg(k,16,  sizeof(int),    &do_relu);

    // 7) 启动 kernel
    size_t g[3] = {(size_t)out_c, (size_t)out_h, (size_t)out_w};
    err = clEnqueueNDRangeKernel(q, k, 3, NULL, g, NULL, 0, NULL, NULL);
    clFinish(q);

    // 8) 读回 + 参考对照
    clEnqueueReadBuffer(q, buf_out, CL_TRUE, 0, sizeof(float)*out_sz, out_cl, 0, NULL, NULL);
    conv2d_reference(input, weight, bias, out_ref,
                     in_c, in_h, in_w, out_c, k_h, k_w,
                     out_h, out_w, s_h, s_w, p_h, p_w, do_relu);

    // 9) 误差统计
    double bad_ratio = compare_and_report("CONV2D", out_cl, out_ref, out_sz, 1e-4, 1e-5);

    // 10) 线程数量
    size_t threads = g[0]*g[1]*g[2];
    printf("\n使用 thread 数 (global work-items)：%zu (=%d*%d*%d)\n",
           threads, out_c, out_h, out_w);

    // 11) 清理
    clReleaseMemObject(buf_in); clReleaseMemObject(buf_w);
    clReleaseMemObject(buf_b);  clReleaseMemObject(buf_out);
    clReleaseKernel(k); clReleaseProgram(prog);
    clReleaseCommandQueue(q); clReleaseContext(ctx);
    free(input); free(weight); free(bias); free(out_cl); free(out_ref);

    printf("\n误差比例: %.6f%%\n", bad_ratio);
    return 0;
}
