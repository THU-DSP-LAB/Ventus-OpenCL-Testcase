// main.c — 单 work-group OpenCL SGEMM 测试（带 profiling + 二进制导出 + 校验）
// 构建：gcc -std=c11 -O2 main.c -lOpenCL -o sgemm_1wg
// 运行：./sgemm_1wg
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ── 可改的矩阵尺寸（C = A[MxK] * B[KxN]） ───────────────────────── */
#define M 128/4
#define N 128/4
#define K 128/4

/* ── 单 work-group 内部的 tile 配置：global==local -> 只有 1 个 work-group ─ */
#define TILE_M 1//16
#define TILE_N 1//16
#define TILE_K 1//16

/* ── 小工具 ───────────────────────── */
static void die(const char* msg, cl_int err){
    fprintf(stderr, "%s (err=%d)\n", msg, err);
    exit(1);
}
static void float_to_hex_string(float f,char*buf){
    uint32_t u; memcpy(&u,&f,sizeof(f)); sprintf(buf,"h%08x",u);
}
/* 计时工具：返回事件持续时间（毫秒） */
static double event_ms(cl_event evt) {
    cl_ulong t0=0, t1=0;
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, NULL);
    return (t1 - t0) / 1.0e6;
}
/* 导出 program 的设备二进制（单设备场景） */
static const char* dump_program_binary(cl_program prog) {
    static char outname[256];

    // 查询设备个数（本例一般是1）
    cl_uint num_devs = 0;
    clGetProgramInfo(prog, CL_PROGRAM_NUM_DEVICES, sizeof(num_devs), &num_devs, NULL);
    if (num_devs == 0) { snprintf(outname,sizeof(outname),"kernel.bin"); return outname; }

    // 读取每个设备的 binary size（这里只取第一个设备）
    size_t *sizes = (size_t*)calloc(num_devs, sizeof(size_t));
    clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(size_t)*num_devs, sizes, NULL);
    if (sizes[0] == 0) { free(sizes); snprintf(outname,sizeof(outname),"kernel.bin"); return outname; }

    // 读取 binary
    unsigned char **bins = (unsigned char**)calloc(num_devs, sizeof(unsigned char*));
    bins[0] = (unsigned char*)malloc(sizes[0]);
    clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(unsigned char*)*num_devs, bins, NULL);

    // 魔数判断
    const unsigned char *bin = bins[0];
    size_t sz = sizes[0];
    int is_elf   = (sz >= 4 && bin[0]==0x7F && bin[1]=='E' && bin[2]=='L' && bin[3]=='F');      // cubin
    int is_spirv = (sz >= 4 && bin[0]==0x03 && bin[1]==0x02 && bin[2]==0x23 && bin[3]==0x07);  // 0x07230203

    if (is_elf) {
        snprintf(outname, sizeof(outname), "kernel.cubin");
    } else if (is_spirv) {
        snprintf(outname, sizeof(outname), "kernel.spv");
    } else {
        // 在 NVIDIA OpenCL 上，这里通常就是 PTX 文本
        snprintf(outname, sizeof(outname), "kernel.ptx");
    }

    FILE *fp = fopen(outname, "wb"); fwrite(bin, 1, sz, fp); fclose(fp);
    fprintf(stderr, "[INFO] 导出设备二进制 -> %s (size=%zu bytes)\n", outname, sz);

    free(bins[0]); free(bins); free(sizes);
    return outname;
}

/* 打印并统计误差（和 CPU 参考对比） */
static void compare_and_report(const char*tag,const float*cl,const float*ref,int n){
    int bad=0;
    printf("\n--- %s (%d elements) ---\n",tag,n);
    printf("idx\tOpenCL\t\tCPU\t\tΔ%%\tHex(OpenCL)\n");
    int preview = (n<10? n:10);
    for(int i=0;i<n;++i){
        double rel = (ref[i]==0)? 0.0 : (cl[i]-ref[i])/ref[i]*100.0;
        if(i<preview){
            char hex[11]; float_to_hex_string(cl[i],hex);
            printf("%4d\t%+8.5f\t%+8.5f\t%6.2f\t%s\n",i,cl[i],ref[i],rel,hex);
        }
        if(fabs(rel)>1e-1) ++bad;   /* 误差阈值 0.01% */
    }
    if(n>preview) puts("...(省略)");
    printf("不符合阈值元素: %d / %d (%.2f%%)\n",bad,n,100.0*bad/n);
}

int main(void){
    cl_int err = 0;

    /* 1) 平台 / 设备 */
    cl_platform_id plat; err = clGetPlatformIDs(1,&plat,NULL);
    cl_device_id   dev;  err|= clGetDeviceIDs(plat,CL_DEVICE_TYPE_DEFAULT,1,&dev,NULL);
    if(err) die("获取平台/设备失败", err);

    /* 2) 上下文 / 队列：开启 profiling */
    cl_context ctx = clCreateContext(NULL,1,&dev,NULL,NULL,&err);
    if(err || !ctx) die("创建 context 失败", err);
#if CL_TARGET_OPENCL_VERSION >= 200
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx,dev,props,&err);
#else
    cl_command_queue q = clCreateCommandQueue(ctx,dev,CL_QUEUE_PROFILING_ENABLE,&err);
#endif
    if(err || !q) die("创建 command queue 失败", err);

    /* 3) 编译 program / kernel */
    const char *kernel_file = "matmul.cl";
    FILE *fp=fopen(kernel_file,"rb");
    if(!fp) { fprintf(stderr,"无法打开 %s\n", kernel_file); return 1; }
    fseek(fp,0,SEEK_END); long sz=ftell(fp); rewind(fp);
    char *src=(char*)malloc(sz+1); fread(src,1,sz,fp); src[sz]='\0'; fclose(fp);

    cl_program prog=clCreateProgramWithSource(ctx,1,(const char**)&src,NULL,&err);
    free(src);
    if(err || !prog) die("创建 program 失败", err);

    char buildopts[256];
    snprintf(buildopts,sizeof(buildopts),
             "-cl-mad-enable -cl-fast-relaxed-math "
             "-DTILE_M=%d -DTILE_N=%d -DTILE_K=%d", TILE_M, TILE_N, TILE_K);

    err = clBuildProgram(prog,1,&dev,buildopts,NULL,NULL);
    if(err){
        size_t logsz=0; clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,0,NULL,&logsz);
        char *log=(char*)malloc(logsz+1);
        clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,logsz,log,NULL); log[logsz]='\0';
        fprintf(stderr,"[Build Log]\n%s\n",log); free(log);
        die("编译失败", err);
    }
    const char* dump = dump_program_binary(prog);

    cl_kernel k=clCreateKernel(prog,"sgemm_one_wg",&err);
    if(err || !k) die("创建 kernel 失败", err);

    /* 4) 准备数据（生成可复现的 A、B；CPU 做参考） */
    float *A=(float*)malloc(sizeof(float)*M*K);
    float *B=(float*)malloc(sizeof(float)*K*N);
    float *C=(float*)malloc(sizeof(float)*M*N);
    float *C_ref=(float*)malloc(sizeof(float)*M*N);

    for(int i=0;i<M*K;++i) A[i] = 0.5f*sinf(i*0.01f) + 0.5f*cosf(i*0.013f);
    for(int i=0;i<K*N;++i) B[i] = sinf(i*0.02f) - 0.25f*cosf(i*0.017f);

    // CPU 参考
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            float acc=0.0f;
            for(int kk=0;kk<K;kk++) acc += A[i*K+kk]*B[kk*N+j];
            C_ref[i*N+j]=acc;
        }
    }

    /* 5) OpenCL 缓冲区 */
    cl_mem bufA=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(float)*M*K,A,&err);
    cl_mem bufB=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(float)*K*N,B,&err);
    cl_mem bufC=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,sizeof(float)*M*N,NULL,&err);
    if(!bufA||!bufB||!bufC) die("创建 buffer 失败", err);

    /* 6) 设置参数 */
    err  = clSetKernelArg(k,0,sizeof(cl_mem),&bufA);
    err |= clSetKernelArg(k,1,sizeof(cl_mem),&bufB);
    err |= clSetKernelArg(k,2,sizeof(cl_mem),&bufC);
    int m=M,n=N,kk=K;
    err |= clSetKernelArg(k,3,sizeof(int),&m);
    err |= clSetKernelArg(k,4,sizeof(int),&n);
    err |= clSetKernelArg(k,5,sizeof(int),&kk);
    if(err) die("设置 kernel 参数失败", err);

    /* 7) 只发起一次 NDRange：global == local -> 只有 1 个 work-group */
    size_t global[2] = { TILE_N, TILE_M };
    size_t local [2] = { TILE_N, TILE_M };
    cl_event evt;
    err = clEnqueueNDRangeKernel(q,k,2,NULL,global,local,0,NULL,&evt);
    if(err) die("Enqueue kernel 失败", err);
    clFinish(q);

    /* 8) 计时 + 性能 */
    double ms = event_ms(evt);
    double gflops = (2.0*(double)M*(double)N*(double)K) / (ms*1.0e6);
    printf("\n=== SGEMM (单 work-group) Profiling ===\n");
    printf("size: M=%d N=%d K=%d | time: %.3f ms | perf: %.2f GFLOP/s\n", M,N,K,ms,gflops);
    printf("[INFO] 已导出设备二进制：%s\n", dump);

    /* 9) 读回 & 校验 */
    err = clEnqueueReadBuffer(q,bufC,CL_TRUE,0,sizeof(float)*M*N,C,0,NULL,NULL);
    if(err) die("ReadBuffer 失败", err);

    compare_and_report("C (OpenCL vs CPU)", C, C_ref, M*N);

    /* 10) 统计 thread / work-group 数量 */
    size_t num_threads = global[0]*global[1];
    size_t num_wg = (global[0]/local[0])*(global[1]/local[1]); //=1
    printf("\n线程数 (global work-items): %zu\n", num_threads);
    printf("work-group 数量          : %zu\n", num_wg);

    /* 11) 资源释放 */
    clReleaseEvent(evt);
    clReleaseMemObject(bufA); clReleaseMemObject(bufB); clReleaseMemObject(bufC);
    clReleaseKernel(k); clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx);
    free(A); free(B); free(C); free(C_ref);
    return 0;
}
