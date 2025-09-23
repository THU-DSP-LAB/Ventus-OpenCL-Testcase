// main.c  –  host code with per-layer check & block count + profiling + binary dump
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <ctype.h>
/* ── 网络超参 ───────────────────────── */
#define IN1_CHANNELS 1
#define IN1_H 28
#define IN1_W 28
#define CONV1_OUT_CHANNELS 2
#define CONV1_K 5
#define CONV1_STRIDE 1
#define CONV1_OUT_H (IN1_H - CONV1_K + 1)   /* 24 */
#define CONV1_OUT_W (IN1_W - CONV1_K + 1)   /* 24 */

#define CONV2_OUT_CHANNELS 1
#define CONV2_K 5
#define CONV2_STRIDE 5
#define CONV2_IN_CHANNELS  CONV1_OUT_CHANNELS
#define CONV2_IN_H         CONV1_OUT_H
#define CONV2_IN_W         CONV1_OUT_W
#define CONV2_OUT_H ((CONV2_IN_H - CONV2_K) / CONV2_STRIDE + 1) /* 4 */
#define CONV2_OUT_W ((CONV2_IN_W - CONV2_K) / CONV2_STRIDE + 1) /* 4 */

#define CONV3_OUT_CHANNELS 10
#define CONV3_K 4
#define CONV3_STRIDE 1
#define CONV3_IN_CHANNELS  CONV2_OUT_CHANNELS
#define CONV3_IN_H         CONV2_OUT_H
#define CONV3_IN_W         CONV2_OUT_W
#define CONV3_OUT_H 1
#define CONV3_OUT_W 1

/* ── 工具函数 ───────────────────────── */
static float hex_to_float(const char *hexstr){
    uint32_t u=(uint32_t)strtoul(hexstr+1,NULL,16);
    float f; memcpy(&f,&u,sizeof(f)); return f;
}
static void float_to_hex_string(float f,char*buf){
    uint32_t u; memcpy(&u,&f,sizeof(f)); sprintf(buf,"h%08x",u);
}
static float* load_array_from_hex(const char*file,int*cnt){
    FILE*fp=fopen(file,"r"); if(!fp){fprintf(stderr,"打开%s失败\n",file);exit(1);}
    fseek(fp,0,SEEK_END); long sz=ftell(fp); rewind(fp);
    char*data=(char*)malloc(sz+1); fread(data,1,sz,fp); data[sz]='\0'; fclose(fp);
    int n=1; for(char*p=data;*p;++p) if(*p==' ') ++n;
    float*arr=(float*)malloc(n*sizeof(float));
    int i=0; for(char*p=strtok(data," \n");p;p=strtok(NULL," \n")) arr[i++]=hex_to_float(p);
    *cnt=n; free(data); return arr;
}
static char* load_kernel_source(const char*file){
    FILE*fp=fopen(file,"r"); if(!fp){fprintf(stderr,"无法打开%s\n",file);exit(1);}
    fseek(fp,0,SEEK_END); long sz=ftell(fp); rewind(fp);
    char*src=(char*)malloc(sz+1); fread(src,1,sz,fp); src[sz]='\0'; fclose(fp);
    return src;
}
/* 打印并统计误差 */
static void compare_and_report(const char*tag,const float*cl,const float*ref,int n){
    int bad=0;
    printf("\n--- %s (%d elements) ---\n",tag,n);
    printf("idx\tOpenCL\t\tPython\t\tΔ%%\tHex(OpenCL)\n");
    int preview = (n<10? n:10);
    for(int i=0;i<n;++i){
        double rel = (ref[i]==0)? 0.0 : (cl[i]-ref[i])/ref[i]*100.0;
        if(i<preview){
            char hex[11]; float_to_hex_string(cl[i],hex);
            printf("%4d\t%+8.5f\t%+8.5f\t%6.2f\t%s\n",i,cl[i],ref[i],rel,hex);
        }
        if(fabs(rel)>1e-2) ++bad;   /* 误差阈值 0.01% */
    }
    if(n>preview) puts("...(省略)");
    printf("不符合阈值元素: %d / %d (%.2f%%)\n",bad,n,100.0*bad/n);
}

/* 计时工具：返回事件持续时间（毫秒） */
static double event_ms(cl_event evt) {
    cl_ulong t0=0, t1=0;
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, NULL);
    return (t1 - t0) / 1.0e6;
}

/* 将 OpenCL program 的设备二进制导出到文件，返回写入的文件名（静态缓冲，便于打印） */
static const char* dump_program_binary(cl_program prog) {
    static char outname[256];
    size_t sz = 0;
    clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &sz, NULL);
    unsigned char *bin = (unsigned char*)malloc(sz);
    unsigned char *bins[] = { bin };
    clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(bins), bins, NULL);

    /* 判定类型：ELF(0x7F,'E','L','F') -> cubin；否则按 PTX 文本保存 */
    if (sz >= 4 && bin[0]==0x7f && bin[1]=='E' && bin[2]=='L' && bin[3]=='F') {
        snprintf(outname, sizeof(outname), "kernel_sm89.cubin");
        FILE *fp = fopen(outname, "wb"); fwrite(bin, 1, sz, fp); fclose(fp);
        fprintf(stderr, "[INFO] 导出 cubin -> %s (size=%zu bytes)\n", outname, sz);
    } else {
        snprintf(outname, sizeof(outname), "kernel.ptx");
        FILE *fp = fopen(outname, "wb"); fwrite(bin, 1, sz, fp); fclose(fp);
        fprintf(stderr, "[INFO] 导出 PTX -> %s (size=%zu bytes)\n", outname, sz);
    }
    free(bin);
    return outname;
}

int main(void){
    cl_int err = 0;

    /* 1) 平台 / 设备（保持不变） */
    cl_platform_id plat; err = clGetPlatformIDs(1,&plat,NULL);
    cl_device_id   dev;  err|= clGetDeviceIDs(plat,CL_DEVICE_TYPE_DEFAULT,1,&dev,NULL);

    /* 2) 上下文 / 队列：开启 profiling */
    cl_context ctx = clCreateContext(NULL,1,&dev,NULL,NULL,&err);
    cl_command_queue q = clCreateCommandQueue(ctx,dev,CL_QUEUE_PROFILING_ENABLE,&err);
    // 如果你的 OpenCL 是 2.0+，也可用 clCreateCommandQueueWithProperties 开启 CL_QUEUE_PROFILING_ENABLE

    /* 3) program / kernel */
    char*src=load_kernel_source("conv.cl");
    cl_program prog=clCreateProgramWithSource(ctx,1,(const char**)&src,NULL,&err);
    free(src);
    // 你也可添加 NVIDIA 的编译 log： -cl-nv-verbose
    err|=clBuildProgram(prog,1,&dev,NULL,NULL,NULL);
    // 导出二进制（PTX 或 CUBIN），供后续 SASS 使用
    const char *binfile = dump_program_binary(prog);

    cl_kernel k=clCreateKernel(prog,"conv",&err);

/* 4) 读权重 & 输入 */
    int c; float *w1=load_array_from_hex("data_gen/conv1_weight.txt",&c);
    float *b1=load_array_from_hex("data_gen/conv1_bias.txt",&c);
    float *w2=load_array_from_hex("data_gen/conv2_weight.txt",&c);
    float *b2=load_array_from_hex("data_gen/conv2_bias.txt",&c);
    float *w3=load_array_from_hex("data_gen/conv3_weight.txt",&c);
    float *b3=load_array_from_hex("data_gen/conv3_bias.txt",&c);
    float *in =load_array_from_hex("data_gen/test_input.txt",&c);

/* 5) 缓冲区 */
    cl_mem buf_in =clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        sizeof(float)*IN1_CHANNELS*IN1_H*IN1_W,in,&err);
    cl_mem buf1 =clCreateBuffer(ctx,CL_MEM_READ_WRITE,
        sizeof(float)*CONV1_OUT_CHANNELS*CONV1_OUT_H*CONV1_OUT_W,NULL,&err);
    cl_mem buf2 =clCreateBuffer(ctx,CL_MEM_READ_WRITE,
        sizeof(float)*CONV2_OUT_CHANNELS*CONV2_OUT_H*CONV2_OUT_W,NULL,&err);
    cl_mem buf3 =clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,
        sizeof(float)*CONV3_OUT_CHANNELS,NULL,&err);

    cl_mem bw1=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        sizeof(float)*CONV1_OUT_CHANNELS*IN1_CHANNELS*CONV1_K*CONV1_K,w1,&err);
    cl_mem bb1=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        sizeof(float)*CONV1_OUT_CHANNELS,b1,&err);
    cl_mem bw2=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        sizeof(float)*CONV2_OUT_CHANNELS*CONV1_OUT_CHANNELS*CONV2_K*CONV2_K,w2,&err);
    cl_mem bb2=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        sizeof(float)*CONV2_OUT_CHANNELS,b2,&err);
    cl_mem bw3=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        sizeof(float)*CONV3_OUT_CHANNELS*CONV2_OUT_CHANNELS*CONV3_K*CONV3_K,w3,&err);
    cl_mem bb3=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        sizeof(float)*CONV3_OUT_CHANNELS,b3,&err);

/* 6) 宏封装设置参数 */
#define SET_ARGS(in,w,b,out,IC,IH,IW,KH,KW,OH,OW,RELU,SH,SW)                \
    do{                                                                     \
        int _ic=IC,_ih=IH,_iw=IW,_kh=KH,_kw=KW,_oh=OH,_ow=OW,_relu=RELU;    \
        int _sh=SH,_sw=SW;                                                  \
        err  = clSetKernelArg(k,0,sizeof(cl_mem),&(in));                    \
        err |= clSetKernelArg(k,1,sizeof(cl_mem),&(w));                     \
        err |= clSetKernelArg(k,2,sizeof(cl_mem),&(b));                     \
        err |= clSetKernelArg(k,3,sizeof(cl_mem),&(out));                   \
        err |= clSetKernelArg(k,4,sizeof(int),&_ic);                        \
        err |= clSetKernelArg(k,5,sizeof(int),&_ih);                        \
        err |= clSetKernelArg(k,6,sizeof(int),&_iw);                        \
        err |= clSetKernelArg(k,7,sizeof(int),&_kh);                        \
        err |= clSetKernelArg(k,8,sizeof(int),&_kw);                        \
        err |= clSetKernelArg(k,9,sizeof(int),&_oh);                        \
        err |= clSetKernelArg(k,10,sizeof(int),&_ow);                       \
        err |= clSetKernelArg(k,11,sizeof(int),&_relu);                     \
        err |= clSetKernelArg(k,12,sizeof(int),&_sh);                       \
        err |= clSetKernelArg(k,13,sizeof(int),&_sw);                       \
    }while(0)

    /* -------- conv1 -------- */
    SET_ARGS(buf_in,bw1,bb1,buf1,
             IN1_CHANNELS,IN1_H,IN1_W,CONV1_K,CONV1_K,
             CONV1_OUT_H,CONV1_OUT_W,
             1,CONV1_STRIDE,CONV1_STRIDE);
    size_t g1[3]={CONV1_OUT_CHANNELS,CONV1_OUT_H,CONV1_OUT_W};
    cl_event e1;
    err|=clEnqueueNDRangeKernel(q,k,3,NULL,g1,NULL,0,NULL,&e1);

    /* -------- conv2 -------- */
    SET_ARGS(buf1,bw2,bb2,buf2,
             CONV2_IN_CHANNELS,CONV2_IN_H,CONV2_IN_W,CONV2_K,CONV2_K,
             CONV2_OUT_H,CONV2_OUT_W,
             1,CONV2_STRIDE,CONV2_STRIDE);
    size_t g2[3]={CONV2_OUT_CHANNELS,CONV2_OUT_H,CONV2_OUT_W};
    cl_event e2;
    err|=clEnqueueNDRangeKernel(q,k,3,NULL,g2,NULL,0,NULL,&e2);

    /* -------- conv3 -------- */
    SET_ARGS(buf2,bw3,bb3,buf3,
             CONV3_IN_CHANNELS,CONV3_IN_H,CONV3_IN_W,CONV3_K,CONV3_K,
             CONV3_OUT_H,CONV3_OUT_W,
             0,CONV3_STRIDE,CONV3_STRIDE);
    size_t g3[3]={CONV3_OUT_CHANNELS,1,1};
    cl_event e3;
    err|=clEnqueueNDRangeKernel(q,k,3,NULL,g3,NULL,0,NULL,&e3);

    clFinish(q); // 确保全部完成

    /* === 计时统计（纳秒级 -> 毫秒） === */
    double ms1 = event_ms(e1);
    double ms2 = event_ms(e2);
    double ms3 = event_ms(e3);
    // 总持续：用第一个 start 到最后一个 end
    cl_ulong s0=0, s1=0, s2=0, s3=0;
    clGetEventProfilingInfo(e1, CL_PROFILING_COMMAND_START, sizeof(s0), &s0, NULL);
    clGetEventProfilingInfo(e3, CL_PROFILING_COMMAND_END,   sizeof(s3), &s3, NULL);
    double ms_total = (s3 - s0) / 1.0e6;

    printf("\n=== Kernel Timings (OpenCL event, ms) ===\n");
    printf("conv1: %.3f ms | conv2: %.3f ms | conv3: %.3f ms | total: %.3f ms\n",
           ms1, ms2, ms3, ms_total);

/* 7) 读回每层输出 */
    float out1[CONV1_OUT_CHANNELS*CONV1_OUT_H*CONV1_OUT_W];
    float out2[CONV2_OUT_CHANNELS*CONV2_OUT_H*CONV2_OUT_W];
    float out3[CONV3_OUT_CHANNELS];

    clEnqueueReadBuffer(q,buf1,CL_TRUE,0,sizeof(out1),out1,0,NULL,NULL);
    clEnqueueReadBuffer(q,buf2,CL_TRUE,0,sizeof(out2),out2,0,NULL,NULL);
    clEnqueueReadBuffer(q,buf3,CL_TRUE,0,sizeof(out3),out3,0,NULL,NULL);

/* 8) 加载参考输出并比对 */
    int n1,n2,n3;
    float *ref1=load_array_from_hex("data_gen/conv1_out.txt",&n1);
    float *ref2=load_array_from_hex("data_gen/conv2_out.txt",&n2);
    float *ref3=load_array_from_hex("data_gen/conv3_out.txt",&n3);

    compare_and_report("CONV1",out1,ref1,n1);
    compare_and_report("CONV2",out2,ref2,n2);
    compare_and_report("CONV3",out3,ref3,n3);

/* 9) block 数量 */
    size_t blocks1=g1[0]*g1[1]*g1[2];
    size_t blocks2=g2[0]*g2[1]*g2[2];
    size_t blocks3=g3[0]*g3[1]*g3[2];
    printf("\n各层使用 thread 数 (global work-item)：\n");
    printf("conv1 : %zu\nconv2 : %zu\nconv3 : %zu\n"
           "总计  : %zu\n",
           blocks1,blocks2,blocks3,
           blocks1+blocks2+blocks3);

/* 10) 资源释放 */
    clReleaseMemObject(buf_in); clReleaseMemObject(buf1); clReleaseMemObject(buf2); clReleaseMemObject(buf3);
    clReleaseMemObject(bw1); clReleaseMemObject(bb1); clReleaseMemObject(bw2); clReleaseMemObject(bb2);
    clReleaseMemObject(bw3); clReleaseMemObject(bb3);
    clReleaseKernel(k); clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx);
    clReleaseEvent(e1); clReleaseEvent(e2); clReleaseEvent(e3);

    free(w1);free(b1);free(w2);free(b2);free(w3);free(b3);free(in);
    free(ref1);free(ref2);free(ref3);
    return 0;
}
