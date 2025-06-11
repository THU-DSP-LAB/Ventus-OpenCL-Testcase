// main.c  –  host code for new ConvNN (1→2→1→10) with stride support
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ─────────────────────────────
 *  网络结构（与 PyTorch 一致）
 * ────────────────────────────*/
#define IN1_CHANNELS 1
#define IN1_H        28
#define IN1_W        28

/* conv1: 1→2, k=5, s=1  →  [2,24,24] */
#define CONV1_OUT_CHANNELS 2
#define CONV1_K            5
#define CONV1_STRIDE       1
#define CONV1_OUT_H        (IN1_H - CONV1_K + 1)   /* 24 */
#define CONV1_OUT_W        (IN1_W - CONV1_K + 1)   /* 24 */

/* conv2: 2→1, k=5, s=5  →  [1,4,4]  */
#define CONV2_OUT_CHANNELS 1
#define CONV2_K            5
#define CONV2_STRIDE       5
#define CONV2_IN_CHANNELS  CONV1_OUT_CHANNELS
#define CONV2_IN_H         CONV1_OUT_H
#define CONV2_IN_W         CONV1_OUT_W
#define CONV2_OUT_H        ((CONV2_IN_H - CONV2_K) / CONV2_STRIDE + 1) /* 4 */
#define CONV2_OUT_W        ((CONV2_IN_W - CONV2_K) / CONV2_STRIDE + 1) /* 4 */

/* conv3: 1→10, k=4, s=1 →  [10,1,1] */
#define CONV3_OUT_CHANNELS 10
#define CONV3_K            4
#define CONV3_STRIDE       1
#define CONV3_IN_CHANNELS  CONV2_OUT_CHANNELS
#define CONV3_IN_H         CONV2_OUT_H
#define CONV3_IN_W         CONV2_OUT_W
#define CONV3_OUT_H        1
#define CONV3_OUT_W        1

/* ─────────── 工具函数 ─────────── */
static float hex_to_float(const char *hexstr) {
    uint32_t u = (uint32_t)strtoul(hexstr + 1, NULL, 16); /* 跳过前缀 'h' */
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static void float_to_hex_string(float f, char *buffer) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(f));
    sprintf(buffer, "h%08x", bits);
}

static float* load_array_from_hex(const char *filename, int *count) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "无法打开文件 %s\n", filename); exit(1); }

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    rewind(fp);

    char *data = (char*)malloc(fsize + 1);
    fread(data, 1, fsize, fp);
    data[fsize] = '\0';
    fclose(fp);

    int cnt = 1;
    for (char *p = data; *p; ++p) if (*p == ' ') ++cnt;

    float *arr = (float*)malloc(cnt * sizeof(float));
    int idx = 0;
    char *tok = strtok(data, " \n");
    while (tok) {
        arr[idx++] = hex_to_float(tok);
        tok = strtok(NULL, " \n");
    }
    *count = cnt;
    free(data);
    return arr;
}

static char* load_kernel_source(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "无法打开 kernel 文件 %s\n", filename); exit(1); }
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    rewind(fp);
    char *src = (char*)malloc(size + 1);
    fread(src, 1, size, fp);
    src[size] = '\0';
    fclose(fp);
    return src;
}

/* ─────────────────────────────
 *              main
 * ────────────────────────────*/
int main(void) {
    cl_int err;

/* 1.  平台 / 设备 */
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err) { puts("clGetPlatformIDs 出错"); return -1; }

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
    if (err) { puts("clGetDeviceIDs 出错"); return -1; }

/* 2.  上下文和队列 */
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err) { puts("clCreateContext 出错"); return -1; }

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err) { puts("clCreateCommandQueue 出错"); return -1; }

/* 3.  编译 kernel（conv.cl 需含 stride_h/stride_w 参数） */
    char *src = load_kernel_source("conv.cl");
    cl_program program = clCreateProgramWithSource(context, 1,
                              (const char**)&src, NULL, &err);
    free(src);
    if (err) { puts("clCreateProgramWithSource 出错"); return -1; }

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err) {
        size_t log_size; clGetProgramBuildInfo(program, device,
                        CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device,
                        CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("编译失败:\n%s\n", log);
        free(log); return -1;
    }

    cl_kernel conv_kernel = clCreateKernel(program, "conv", &err);
    if (err) { puts("创建 kernel 失败"); return -1; }

/* 4.  加载权重 / 输入 */
    int cnt;
    float *w1 = load_array_from_hex("./data_gen/conv1_weight.txt", &cnt); /* 2*1*5*5 */
    float *b1 = load_array_from_hex("./data_gen/conv1_bias.txt",   &cnt); /* 2 */
    float *w2 = load_array_from_hex("./data_gen/conv2_weight.txt", &cnt); /* 1*2*5*5 */
    float *b2 = load_array_from_hex("./data_gen/conv2_bias.txt",   &cnt); /* 1 */
    float *w3 = load_array_from_hex("./data_gen/conv3_weight.txt", &cnt); /* 10*1*4*4 */
    float *b3 = load_array_from_hex("./data_gen/conv3_bias.txt",   &cnt); /* 10 */
    float *test_in = load_array_from_hex("./data_gen/test_input.txt", &cnt); /* 28*28 */

/* 5.  创建缓冲区 */
    cl_mem buf_in  = clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
                       sizeof(float)*IN1_CHANNELS*IN1_H*IN1_W, test_in, &err);

    cl_mem buf_c1o = clCreateBuffer(context, CL_MEM_READ_WRITE,
                       sizeof(float)*CONV1_OUT_CHANNELS*CONV1_OUT_H*CONV1_OUT_W, NULL, &err);
    cl_mem buf_c2o = clCreateBuffer(context, CL_MEM_READ_WRITE,
                       sizeof(float)*CONV2_OUT_CHANNELS*CONV2_OUT_H*CONV2_OUT_W, NULL, &err);
    cl_mem buf_c3o = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                       sizeof(float)*CONV3_OUT_CHANNELS, NULL, &err);

    cl_mem buf_w1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float)*CONV1_OUT_CHANNELS*IN1_CHANNELS*CONV1_K*CONV1_K, w1, &err);
    cl_mem buf_b1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float)*CONV1_OUT_CHANNELS, b1, &err);

    cl_mem buf_w2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float)*CONV2_OUT_CHANNELS*CONV1_OUT_CHANNELS*CONV2_K*CONV2_K, w2, &err);
    cl_mem buf_b2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float)*CONV2_OUT_CHANNELS, b2, &err);

    cl_mem buf_w3 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float)*CONV3_OUT_CHANNELS*CONV2_OUT_CHANNELS*CONV3_K*CONV3_K, w3, &err);
    cl_mem buf_b3 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float)*CONV3_OUT_CHANNELS, b3, &err);

/* 6.  依次调用 conv kernel (14 参数) */
#define SET_COMMON_ARGS(buf_in,buf_w,buf_b,buf_out,IC,IH,IW,KH,KW,OH,OW,ReLU,S_H,S_W) do {  \
        err  = clSetKernelArg(conv_kernel, 0,  sizeof(cl_mem), &(buf_in));   \
        err |= clSetKernelArg(conv_kernel, 1,  sizeof(cl_mem), &(buf_w));    \
        err |= clSetKernelArg(conv_kernel, 2,  sizeof(cl_mem), &(buf_b));    \
        err |= clSetKernelArg(conv_kernel, 3,  sizeof(cl_mem), &(buf_out));  \
        int _ic=IC,_ih=IH,_iw=IW,_kh=KH,_kw=KW,_oh=OH,_ow=OW,_relu=ReLU,_sh=S_H,_sw=S_W; \
        err |= clSetKernelArg(conv_kernel, 4,  sizeof(int), &_ic);           \
        err |= clSetKernelArg(conv_kernel, 5,  sizeof(int), &_ih);           \
        err |= clSetKernelArg(conv_kernel, 6,  sizeof(int), &_iw);           \
        err |= clSetKernelArg(conv_kernel, 7,  sizeof(int), &_kh);           \
        err |= clSetKernelArg(conv_kernel, 8,  sizeof(int), &_kw);           \
        err |= clSetKernelArg(conv_kernel, 9,  sizeof(int), &_oh);           \
        err |= clSetKernelArg(conv_kernel,10, sizeof(int), &_ow);           \
        err |= clSetKernelArg(conv_kernel,11, sizeof(int), &_relu);         \
        err |= clSetKernelArg(conv_kernel,12, sizeof(int), &_sh);           \
        err |= clSetKernelArg(conv_kernel,13, sizeof(int), &_sw);           \
    } while(0)

/* ── conv1 ───────────────────────────────────────────────────────── */
    SET_COMMON_ARGS(buf_in, buf_w1, buf_b1, buf_c1o,
                    IN1_CHANNELS, IN1_H, IN1_W,
                    CONV1_K, CONV1_K,
                    CONV1_OUT_H, CONV1_OUT_W,
                    1,  /* ReLU */
                    CONV1_STRIDE, CONV1_STRIDE);

    size_t g1[3] = { CONV1_OUT_CHANNELS, CONV1_OUT_H, CONV1_OUT_W };
    err |= clEnqueueNDRangeKernel(queue, conv_kernel, 3, NULL, g1, NULL, 0, NULL, NULL);
    if (err) { puts("conv1 执行失败"); return -1; }

/* ── conv2 ───────────────────────────────────────────────────────── */
    SET_COMMON_ARGS(buf_c1o, buf_w2, buf_b2, buf_c2o,
                    CONV2_IN_CHANNELS, CONV2_IN_H, CONV2_IN_W,
                    CONV2_K, CONV2_K,
                    CONV2_OUT_H, CONV2_OUT_W,
                    1,  /* ReLU */
                    CONV2_STRIDE, CONV2_STRIDE);

    size_t g2[3] = { CONV2_OUT_CHANNELS, CONV2_OUT_H, CONV2_OUT_W };
    err |= clEnqueueNDRangeKernel(queue, conv_kernel, 3, NULL, g2, NULL, 0, NULL, NULL);
    if (err) { puts("conv2 执行失败"); return -1; }

/* ── conv3 ───────────────────────────────────────────────────────── */
    SET_COMMON_ARGS(buf_c2o, buf_w3, buf_b3, buf_c3o,
                    CONV3_IN_CHANNELS, CONV3_IN_H, CONV3_IN_W,
                    CONV3_K, CONV3_K,
                    CONV3_OUT_H, CONV3_OUT_W,
                    0,  /* no ReLU */
                    CONV3_STRIDE, CONV3_STRIDE);

    size_t g3[3] = { CONV3_OUT_CHANNELS, 1, 1 };
    err |= clEnqueueNDRangeKernel(queue, conv_kernel, 3, NULL, g3, NULL, 0, NULL, NULL);
    if (err) { puts("conv3 执行失败"); return -1; }

/* 7.  等待并读回结果 */
    clFinish(queue);

    float out_host[CONV3_OUT_CHANNELS];
    err = clEnqueueReadBuffer(queue, buf_c3o, CL_TRUE, 0,
                              sizeof(out_host), out_host, 0, NULL, NULL);
    if (err) { puts("读回输出失败"); return -1; }

/* 8.  加载 Python 端基准输出并对比 */
    int ref_cnt;
    float *ref_out = load_array_from_hex("./data_gen/test_output.txt", &ref_cnt);
    if (ref_cnt != CONV3_OUT_CHANNELS) {
        printf("参考输出数量(%d) 与期望(%d) 不符\n", ref_cnt, CONV3_OUT_CHANNELS);
    }

    puts("──────── 推理结果对比 ────────");
    puts("idx\tOpenCL\t\tPython\t\t误差%\thex");
    for (int i = 0; i < CONV3_OUT_CHANNELS; ++i) {
        char hexbuf[11];
        float_to_hex_string(out_host[i], hexbuf);
        double rel = (ref_out[i] == 0.0f) ? 0.0
                     : (out_host[i] - ref_out[i]) / ref_out[i] * 100.0;
        printf("%2d\t%+8.5f\t%+8.5f\t%6.2f\t%s\n",
               i, out_host[i], ref_out[i], rel, hexbuf);
    }

/* 9.  释放资源 */
    clReleaseMemObject(buf_in);   clReleaseMemObject(buf_c1o);
    clReleaseMemObject(buf_c2o);  clReleaseMemObject(buf_c3o);
    clReleaseMemObject(buf_w1);   clReleaseMemObject(buf_b1);
    clReleaseMemObject(buf_w2);   clReleaseMemObject(buf_b2);
    clReleaseMemObject(buf_w3);   clReleaseMemObject(buf_b3);
    clReleaseKernel(conv_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(w1); free(b1); free(w2); free(b2); free(w3); free(b3);
    free(test_in); free(ref_out);

    return 0;
}
