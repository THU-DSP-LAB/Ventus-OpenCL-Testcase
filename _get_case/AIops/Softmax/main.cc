// softmax_main.cc — Host test for softmax_rows / softmax_chw
// 随机输入 -> OpenCL kernel -> CPU reference 校验（数值稳定实现）
// 构建：clang++ -O2 -std=c++11 softmax_main.cc -lOpenCL -o Softmax.out

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <algorithm>

static const char* KERNEL_PATH = "softmax.cl"; // 如需调整路径，改这里

/* 小工具 */
static inline float frand(){ return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }

static char* load_text_file(const char* path, size_t* out_sz=nullptr){
    FILE* fp = fopen(path, "rb");
    if(!fp){ fprintf(stderr,"无法打开 %s\n", path); exit(1); }
    fseek(fp,0,SEEK_END); long sz=ftell(fp); rewind(fp);
    char* buf=(char*)malloc(sz+1);
    fread(buf,1,sz,fp); buf[sz]='\0'; fclose(fp);
    if(out_sz) *out_sz = (size_t)sz;
    return buf;
}

static cl_program build_program(cl_context ctx, cl_device_id dev, const char* path){
    size_t sz=0; char* src=load_text_file(path,&sz);
    cl_int err=CL_SUCCESS;
    const char* p=src;
    cl_program prog = clCreateProgramWithSource(ctx,1,&p,&sz,&err);
    if(err!=CL_SUCCESS){ fprintf(stderr,"clCreateProgramWithSource err=%d\n",err); exit(1); }
    err = clBuildProgram(prog,1,&dev,nullptr,nullptr,nullptr);
    if(err!=CL_SUCCESS){
        size_t logsz=0; clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,0,nullptr,&logsz);
        std::vector<char> log(logsz+1);
        clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,logsz,log.data(),nullptr);
        log[logsz]='\0';
        fprintf(stderr,"Build log:\n%s\n", log.data());
        exit(1);
    }
    free(src);
    return prog;
}

/* ============ CPU 参考实现（数值稳定） ============ */
static void softmax_rows_ref(const float* x, float* y, int M, int N){
    for(int r=0;r<M;++r){
        const float* xr = x + r*N;
        float* yr = y + r*N;
        float m = xr[0];
        for(int j=1;j<N;++j) m = std::max(m, xr[j]);
        double sum = 0.0;
        for(int j=0;j<N;++j){ double e = std::exp((double)xr[j] - (double)m); yr[j]=(float)e; sum += e; }
        double inv = (sum>0.0)? 1.0/sum : 0.0;
        for(int j=0;j<N;++j) yr[j] = (float)((double)yr[j]*inv);
    }
}

static void softmax_chw_ref(const float* x, float* y, int C, int H, int W){
    int HW = H*W;
    for(int yy=0; yy<H; ++yy){
        for(int xx=0; xx<W; ++xx){
            int col = yy*W + xx;
            float m = x[col];
            for(int c=1;c<C;++c) m = std::max(m, x[c*HW + col]);
            double sum = 0.0;
            for(int c=0;c<C;++c){ double e = std::exp((double)x[c*HW+col] - (double)m); y[c*HW+col] = (float)e; sum+=e; }
            double inv = (sum>0.0)? 1.0/sum : 0.0;
            for(int c=0;c<C;++c) y[c*HW+col] = (float)((double)y[c*HW+col]*inv);
        }
    }
}

/* 误差统计（返回不合格比例%） */
static double compare_and_report(const char* tag, const float* got, const float* ref, int n,
                                 double rtol=1e-5, double atol=1e-6){
    int bad=0; double max_abs=0.0, max_rel=0.0;
    int preview = std::min(n, 10);
    printf("\n--- %s (%d elements) ---\n", tag, n);
    printf("idx\tgot\t\tref\t\tabs\t\trel\n");
    for(int i=0;i<n;++i){
        double a = std::fabs((double)got[i]- (double)ref[i]);
        double r = (std::fabs(ref[i])>1e-12)? a/std::fabs(ref[i]) : 0.0;
        if(i<preview) printf("%4d\t%+9.6f\t%+9.6f\t%1.3e\t%1.3e\n", i, got[i], ref[i], a, r);
        if(!(a<=atol || r<=rtol)) ++bad;
        if(a>max_abs) max_abs=a;
        if(r>max_rel) max_rel=r;
    }
    if(n>preview) puts("...(省略)");
    double ratio = 100.0*(double)bad/(double)n;
    printf("不符合阈值元素: %d/%d (%.4f%%), max_abs=%g, max_rel=%g\n", bad,n,ratio,max_abs,max_rel);
    return ratio;
}

int main(int argc, char** argv){
    srand((unsigned)time(NULL));

    // 默认测试规模
    int M=3, N=7;         // rows softmax
    int C=5, H=4, W=3;    // CHW softmax

    // 简单命令行覆盖：--M 1 --N 1000 --C 512 --H 1 --W 1
    for(int i=1;i<argc;++i){
        auto next=[&](int& dst){ if(i+1<argc) dst=atoi(argv[++i]); };
        if(!strcmp(argv[i],"--M")) next(M);
        else if(!strcmp(argv[i],"--N")) next(N);
        else if(!strcmp(argv[i],"--C")) next(C);
        else if(!strcmp(argv[i],"--H")) next(H);
        else if(!strcmp(argv[i],"--W")) next(W);
        else if(!strcmp(argv[i],"--help")||!strcmp(argv[i],"-h")){
            printf("Usage: %s [--M m --N n] [--C c --H h --W w]\n", argv[0]);
            return 0;
        }
    }

    cl_int err=CL_SUCCESS;

    // 平台/设备/上下文/队列
    cl_platform_id plat; cl_device_id dev;
    err = clGetPlatformIDs(1,&plat,NULL);
    err|= clGetDeviceIDs(plat, CL_DEVICE_TYPE_DEFAULT, 1,&dev,NULL);
    cl_context ctx = clCreateContext(NULL,1,&dev,NULL,NULL,&err);
#if defined(CL_VERSION_2_0)
    cl_queue_properties props[] = { 0 };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, props, &err);
#else
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
#endif

    // 编译 softmax.cl
    cl_program prog = build_program(ctx, dev, KERNEL_PATH);
    cl_kernel k_rows = clCreateKernel(prog, "softmax_rows", &err);
    cl_kernel k_chw  = clCreateKernel(prog, "softmax_chw",  &err);

    /* ------------ A) rows softmax 测试 ------------ */
    std::vector<float> x_rows((size_t)M*N), y_rows((size_t)M*N), yref_rows((size_t)M*N);
    for(auto& v: x_rows) v = frand()*5.0f; // 稍微扩大幅度，考验数值稳定
    cl_mem d_xr = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, sizeof(float)*x_rows.size(), x_rows.data(), &err);
    cl_mem d_yr = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float)*y_rows.size(), NULL, &err);

    int arg=0;
    clSetKernelArg(k_rows,arg++,sizeof(cl_mem),&d_xr);
    clSetKernelArg(k_rows,arg++,sizeof(cl_mem),&d_yr);
    clSetKernelArg(k_rows,arg++,sizeof(int),&M);
    clSetKernelArg(k_rows,arg++,sizeof(int),&N);
    size_t g_rows[1] = { (size_t)M };
    clEnqueueNDRangeKernel(q, k_rows, 1, NULL, g_rows, NULL, 0, NULL, NULL);
    clFinish(q);

    clEnqueueReadBuffer(q, d_yr, CL_TRUE, 0, sizeof(float)*y_rows.size(), y_rows.data(), 0, NULL, NULL);

    // CPU 参考
    softmax_rows_ref(x_rows.data(), yref_rows.data(), M, N);

    // 行归一化检查 & 误差报告
    for(int r=0;r<M;++r){
        double s=0.0; for(int j=0;j<N;++j) s += (double)y_rows[r*N+j];
        printf("rows[%d] sum = %.9f (应≈1)\n", r, s);
    }
    compare_and_report("softmax_rows", y_rows.data(), yref_rows.data(), (int)y_rows.size(), 1e-5, 1e-6);

    /* ------------ B) CHW softmax 测试 ------------ */
    std::vector<float> x_chw((size_t)C*H*W), y_chw((size_t)C*H*W), yref_chw((size_t)C*H*W);
    for(auto& v: x_chw) v = frand()*5.0f;
    cl_mem d_xc = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, sizeof(float)*x_chw.size(), x_chw.data(), &err);
    cl_mem d_yc = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float)*y_chw.size(), NULL, &err);

    arg=0;
    clSetKernelArg(k_chw,arg++,sizeof(cl_mem),&d_xc);
    clSetKernelArg(k_chw,arg++,sizeof(cl_mem),&d_yc);
    clSetKernelArg(k_chw,arg++,sizeof(int),&C);
    clSetKernelArg(k_chw,arg++,sizeof(int),&H);
    clSetKernelArg(k_chw,arg++,sizeof(int),&W);
    size_t g_chw[2] = { (size_t)H, (size_t)W };
    clEnqueueNDRangeKernel(q, k_chw, 2, NULL, g_chw, NULL, 0, NULL, NULL);
    clFinish(q);

    clEnqueueReadBuffer(q, d_yc, CL_TRUE, 0, sizeof(float)*y_chw.size(), y_chw.data(), 0, NULL, NULL);

    // CPU 参考
    softmax_chw_ref(x_chw.data(), yref_chw.data(), C, H, W);

    // 每个 (y,x) 的归一化检查
    for(int yy=0; yy<std::min(H,3); ++yy){
        for(int xx=0; xx<std::min(W,3); ++xx){
            double s=0.0;
            for(int c=0;c<C;++c) s += (double)y_chw[c*(H*W) + yy*W + xx];
            printf("chw(y=%d,x=%d) sum = %.9f (应≈1)\n", yy, xx, s);
        }
    }
    compare_and_report("softmax_chw", y_chw.data(), yref_chw.data(), (int)y_chw.size(), 1e-5, 1e-6);

    /* 资源释放 */
    clReleaseMemObject(d_xr); clReleaseMemObject(d_yr);
    clReleaseMemObject(d_xc); clReleaseMemObject(d_yc);
    clReleaseKernel(k_rows); clReleaseKernel(k_chw);
    clReleaseProgram(prog);
#if defined(CL_VERSION_2_0)
    clReleaseCommandQueue(q);
#else
    clReleaseCommandQueue(q);
#endif
    clReleaseContext(ctx);
    return 0;
}
