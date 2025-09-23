#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// 网络各层尺寸配置
// 第一层：conv1: 输入 [1,28,28] → 输出 [16,24,24]，卷积核大小 5x5，激活：ReLU
#define IN1_CHANNELS 1
#define IN1_H 28
#define IN1_W 28

#define CONV1_OUT_CHANNELS 16
#define CONV1_K 5
#define CONV1_OUT_H (IN1_H - CONV1_K + 1)   // 24
#define CONV1_OUT_W (IN1_W - CONV1_K + 1)     // 24

// 第二层：conv2: 输入 [16,24,24] → 输出 [32,20,20]，卷积核大小 5x5，激活：ReLU
#define CONV2_OUT_CHANNELS 32
#define CONV2_K 5
#define CONV2_IN_CHANNELS CONV1_OUT_CHANNELS
#define CONV2_IN_H CONV1_OUT_H   // 24
#define CONV2_IN_W CONV1_OUT_W   // 24
#define CONV2_OUT_H (CONV2_IN_H - CONV2_K + 1)  // 20
#define CONV2_OUT_W (CONV2_IN_W - CONV2_K + 1)  // 20

// 第三层：conv3: 输入 [32,20,20] → 输出 [10,1,1]，卷积核大小 20x20，无激活函数
#define CONV3_OUT_CHANNELS 10
#define CONV3_K 20
#define CONV3_IN_CHANNELS CONV2_OUT_CHANNELS
#define CONV3_IN_H CONV2_OUT_H  // 20
#define CONV3_IN_W CONV2_OUT_W  // 20
// 输出尺寸为 1×1

// 工具函数：将 hex 字符串（例如 "h3f800000"）转换为 float
float hex_to_float(const char *hexstr) {
    uint32_t u = (uint32_t)strtoul(hexstr + 1, NULL, 16); // 跳过前缀 'h'
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// 将 float 转换为十六进制字符串，格式为 "hXXXXXXXX"
// 注意：调用者应确保 buffer 至少有 11 个字符空间（包括结束符）
void float_to_hex_string(float f, char *buffer) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(f));
    sprintf(buffer, "h%08x", bits);
}

// 从文件中读取所有以空格分隔的 hex 格式数据，返回动态分配的 float 数组，并设置 count 为数字个数
float* load_array_from_hex(const char *filename, int *count) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "无法打开文件 %s\n", filename);
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *data = (char*)malloc(fsize + 1);
    fread(data, 1, fsize, fp);
    data[fsize] = '\0';
    fclose(fp);

    int cnt = 0;
    for (char *p = data; *p; p++) {
        if (*p == ' ') cnt++;
    }
    cnt = cnt + 1; // 最后一个数字后可能没有空格

    float *arr = (float*)malloc(cnt * sizeof(float));
    int idx = 0;
    char *token = strtok(data, " \n");
    while (token) {
        arr[idx++] = hex_to_float(token);
        token = strtok(NULL, " \n");
    }
    *count = cnt;
    free(data);
    return arr;
}

static void dump_program_binaries(cl_program prog) {
    cl_int err;
    cl_uint num_devs = 0;
    err = clGetProgramInfo(prog, CL_PROGRAM_NUM_DEVICES, sizeof(num_devs), &num_devs, NULL);
    if (err != CL_SUCCESS || num_devs == 0) { fprintf(stderr, "[ERR] CL_PROGRAM_NUM_DEVICES\n"); return; }

    cl_device_id *devs = (cl_device_id*)calloc(num_devs, sizeof(cl_device_id));
    err = clGetProgramInfo(prog, CL_PROGRAM_DEVICES, sizeof(cl_device_id)*num_devs, devs, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "[ERR] CL_PROGRAM_DEVICES\n"); free(devs); return; }

    size_t *sizes = (size_t*)calloc(num_devs, sizeof(size_t));
    err = clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(size_t)*num_devs, sizes, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "[ERR] CL_PROGRAM_BINARY_SIZES\n"); free(sizes); free(devs); return; }

    unsigned char **bins = (unsigned char**)calloc(num_devs, sizeof(unsigned char*));
    for (cl_uint i=0; i<num_devs; ++i) {
        bins[i] = (unsigned char*)malloc(sizes[i]);
    }
    err = clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(unsigned char*)*num_devs, bins, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "[ERR] CL_PROGRAM_BINARIES\n"); goto cleanup; }

    for (cl_uint i=0; i<num_devs; ++i) {
        const unsigned char *bin = bins[i];
        size_t sz = sizes[i];
        /* 识别类型 */
        int is_elf   = (sz>=4 && bin[0]==0x7F && bin[1]=='E' && bin[2]=='L' && bin[3]=='F');              // cubin
        int is_spirv = (sz>=4 && bin[0]==0x03 && bin[1]==0x02 && bin[2]==0x23 && bin[3]==0x07);          // 0x07230203
        int looks_text_ptx = 0;
        if (!is_elf && !is_spirv) {
            /* 简单判定 PTX 文本：前 256 字节 ASCII 且包含 ".version" */
            size_t scan = sz<256? sz:256;
            looks_text_ptx = 1;
            for (size_t k=0;k<scan;k++){ unsigned char c=bin[k]; if(c<9 || c>126){ looks_text_ptx=0; break; } }
            if (looks_text_ptx) {
                const char *p=(const char*)bin;
                if (!memmem(p, scan, ".version", 8)) looks_text_ptx=0;
            }
        }

        /* 取设备名作为文件名后缀，避免多设备冲突 */
        char devname[128]={0};
        clGetDeviceInfo(devs[i], CL_DEVICE_NAME, sizeof(devname), devname, NULL);
        for (char *p=devname; *p; ++p) if (*p==' '||*p=='/'||*p=='\\'||*p==':') *p='_';

        char outname[256];
        const char* kind = NULL;
        if (is_elf)      { snprintf(outname,sizeof(outname),"kernel_%s.cubin", devname); kind="cubin"; }
        else if (is_spirv){snprintf(outname,sizeof(outname),"kernel_%s.spv",   devname); kind="SPIR-V"; }
        else if (looks_text_ptx){ snprintf(outname,sizeof(outname),"kernel_%s.ptx", devname); kind="PTX(text)"; }
        else             { snprintf(outname,sizeof(outname),"kernel_%s.bin",   devname); kind="unknown(bin)"; }

        FILE *fp=fopen(outname,"wb"); fwrite(bin,1,sz,fp); fclose(fp);
        fprintf(stderr,"[INFO] 导出 %s -> %s (size=%zu bytes)\n", kind, outname, sz);
        /* 方便排查：打印前 8 个字节 */
        fprintf(stderr,"[INFO] head bytes: ");
        for (int b=0;b<8 && b<(int)sz;b++) fprintf(stderr,"%02x ", bin[b]);
        fprintf(stderr,"\n");
    }

cleanup:
    for (cl_uint i=0; i<num_devs; ++i) free(bins[i]);
    free(bins); free(sizes); free(devs);
}

// 加载 kernel 源文件
char* load_kernel_source(const char* filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "无法打开 kernel 文件 %s\n", filename);
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    rewind(fp);
    char *source = (char*)malloc(size + 1);
    fread(source, 1, size, fp);
    source[size] = '\0';
    fclose(fp);
    return source;
}

int main() {
    cl_int err;
    
    // 1. 获取平台与设备
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    if(err != CL_SUCCESS) { printf("clGetPlatformIDs 出错\n"); return -1; }

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
    if(err != CL_SUCCESS) { printf("clGetDeviceIDs 出错\n"); return -1; }
    
    // 2. 创建上下文和命令队列
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if(err != CL_SUCCESS) { printf("clCreateContext 出错\n"); return -1; }

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if(err != CL_SUCCESS) { printf("clCreateCommandQueue 出错\n"); return -1; }
    
    // 3. 加载并编译 kernel 源码（使用通用卷积 kernel）
    char* source = load_kernel_source("conv.cl");
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);
    free(source);
    if(err != CL_SUCCESS) { printf("clCreateProgramWithSource 出错\n"); return -1; }
    
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    // const char *binfile = dump_program_binary(program);
    dump_program_binaries(program);

    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*) malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("编译失败:\n%s\n", log);
        free(log);
        return -1;
    }
    
    // 创建通用卷积 kernel
    cl_kernel conv_kernel = clCreateKernel(program, "conv", &err);
    if(err != CL_SUCCESS) { printf("创建通用 kernel 出错\n"); return -1; }
    
    // 4. 加载各层权重、偏置及测试输入
    int cnt;
    float *conv1_weight = load_array_from_hex("./data_gen/conv1_weight.txt", &cnt); // size: 16*1*5*5
    float *conv1_bias   = load_array_from_hex("./data_gen/conv1_bias.txt", &cnt);   // size: 16
    float *conv2_weight = load_array_from_hex("./data_gen/conv2_weight.txt", &cnt); // size: 32*16*5*5
    float *conv2_bias   = load_array_from_hex("./data_gen/conv2_bias.txt", &cnt);   // size: 32
    float *conv3_weight = load_array_from_hex("./data_gen/conv3_weight.txt", &cnt); // size: 10*32*20*20
    float *conv3_bias   = load_array_from_hex("./data_gen/conv3_bias.txt", &cnt);   // size: 10
    float *test_input   = load_array_from_hex("./data_gen/test_input.txt", &cnt);   // size: 28*28

    // 5. 创建设备缓冲区
    // 输入缓冲区（第一层）：[1,28,28]
    cl_mem buf_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * IN1_CHANNELS * IN1_H * IN1_W, test_input, &err);
    // conv1 输出：[16,24,24]
    cl_mem buf_conv1_out = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(float) * CONV1_OUT_CHANNELS * CONV1_OUT_H * CONV1_OUT_W, NULL, &err);
    // conv2 输出：[32,20,20]
    cl_mem buf_conv2_out = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(float) * CONV2_OUT_CHANNELS * CONV2_OUT_H * CONV2_OUT_W, NULL, &err);
    // conv3 输出：[10,1,1]（展平为 10 个 float）
    cl_mem buf_conv3_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                      sizeof(float) * CONV3_OUT_CHANNELS, NULL, &err);

    // 分别为各层权重和偏置建立缓冲区
    cl_mem buf_conv1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * CONV1_OUT_CHANNELS * IN1_CHANNELS * CONV1_K * CONV1_K,
                                      conv1_weight, &err);
    cl_mem buf_conv1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * CONV1_OUT_CHANNELS,
                                      conv1_bias, &err);

    cl_mem buf_conv2_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * CONV2_OUT_CHANNELS * CONV1_OUT_CHANNELS * CONV2_K * CONV2_K,
                                      conv2_weight, &err);
    cl_mem buf_conv2_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * CONV2_OUT_CHANNELS,
                                      conv2_bias, &err);

    cl_mem buf_conv3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * CONV3_OUT_CHANNELS * CONV2_OUT_CHANNELS * CONV3_K * CONV3_K,
                                      conv3_weight, &err);
    cl_mem buf_conv3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * CONV3_OUT_CHANNELS,
                                      conv3_bias, &err);

    // 6. 依次调用通用卷积 kernel

    // ------ 第一层：conv1 ------
    // 参数设置：in_channels=1, in_h=28, in_w=28, kernel=5x5, out_h=24, out_w=24, do_relu=1
    size_t global_conv1[3] = { CONV1_OUT_CHANNELS, CONV1_OUT_H, CONV1_OUT_W };
    err  = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &buf_input);
    err |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &buf_conv1_weight);
    err |= clSetKernelArg(conv_kernel, 2, sizeof(cl_mem), &buf_conv1_bias);
    err |= clSetKernelArg(conv_kernel, 3, sizeof(cl_mem), &buf_conv1_out);
    int in_channels = IN1_CHANNELS;
    int in_h = IN1_H;
    int in_w = IN1_W;
    int kernel_h = CONV1_K;
    int kernel_w = CONV1_K;
    int out_h = CONV1_OUT_H;
    int out_w = CONV1_OUT_W;
    int do_relu = 1;
    err |= clSetKernelArg(conv_kernel, 4, sizeof(int), &in_channels);
    err |= clSetKernelArg(conv_kernel, 5, sizeof(int), &in_h);
    err |= clSetKernelArg(conv_kernel, 6, sizeof(int), &in_w);
    err |= clSetKernelArg(conv_kernel, 7, sizeof(int), &kernel_h);
    err |= clSetKernelArg(conv_kernel, 8, sizeof(int), &kernel_w);
    err |= clSetKernelArg(conv_kernel, 9, sizeof(int), &out_h);
    err |= clSetKernelArg(conv_kernel, 10, sizeof(int), &out_w);
    err |= clSetKernelArg(conv_kernel, 11, sizeof(int), &do_relu);
    err |= clEnqueueNDRangeKernel(queue, conv_kernel, 3, NULL, global_conv1, NULL, 0, NULL, NULL);
    if(err != CL_SUCCESS) { printf("执行 conv1 出错\n"); return -1; }

    // ------ 第二层：conv2 ------
    // 输入为 conv1 输出：[16,24,24]
    // 参数设置：in_channels=16, in_h=24, in_w=24, kernel=5x5, out_h=20, out_w=20, do_relu=1
    size_t global_conv2[3] = { CONV2_OUT_CHANNELS, CONV2_OUT_H, CONV2_OUT_W };
    err  = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &buf_conv1_out);
    err |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &buf_conv2_weight);
    err |= clSetKernelArg(conv_kernel, 2, sizeof(cl_mem), &buf_conv2_bias);
    err |= clSetKernelArg(conv_kernel, 3, sizeof(cl_mem), &buf_conv2_out);
    in_channels = CONV2_IN_CHANNELS; // 16
    in_h = CONV1_OUT_H;  // 24
    in_w = CONV1_OUT_W;  // 24
    kernel_h = CONV2_K;  // 5
    kernel_w = CONV2_K;  // 5
    out_h = CONV2_OUT_H; // 20
    out_w = CONV2_OUT_W; // 20
    do_relu = 1;
    err |= clSetKernelArg(conv_kernel, 4, sizeof(int), &in_channels);
    err |= clSetKernelArg(conv_kernel, 5, sizeof(int), &in_h);
    err |= clSetKernelArg(conv_kernel, 6, sizeof(int), &in_w);
    err |= clSetKernelArg(conv_kernel, 7, sizeof(int), &kernel_h);
    err |= clSetKernelArg(conv_kernel, 8, sizeof(int), &kernel_w);
    err |= clSetKernelArg(conv_kernel, 9, sizeof(int), &out_h);
    err |= clSetKernelArg(conv_kernel, 10, sizeof(int), &out_w);
    err |= clSetKernelArg(conv_kernel, 11, sizeof(int), &do_relu);
    err |= clEnqueueNDRangeKernel(queue, conv_kernel, 3, NULL, global_conv2, NULL, 0, NULL, NULL);
    if(err != CL_SUCCESS) { printf("执行 conv2 出错\n"); return -1; }

    // ------ 第三层：conv3 ------
    // 输入为 conv2 输出：[32,20,20]
    // 参数设置：in_channels=32, in_h=20, in_w=20, kernel=20x20, out_h=1, out_w=1, do_relu=0
    size_t global_conv3[3] = { CONV3_OUT_CHANNELS, 1, 1 };
    err  = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &buf_conv2_out);
    err |= clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &buf_conv3_weight);
    err |= clSetKernelArg(conv_kernel, 2, sizeof(cl_mem), &buf_conv3_bias);
    err |= clSetKernelArg(conv_kernel, 3, sizeof(cl_mem), &buf_conv3_out);
    in_channels = CONV3_IN_CHANNELS; // 32
    in_h = CONV2_OUT_H;  // 20
    in_w = CONV2_OUT_W;  // 20
    kernel_h = CONV3_K;  // 20
    kernel_w = CONV3_K;  // 20
    out_h = 1;
    out_w = 1;
    do_relu = 0;
    err |= clSetKernelArg(conv_kernel, 4, sizeof(int), &in_channels);
    err |= clSetKernelArg(conv_kernel, 5, sizeof(int), &in_h);
    err |= clSetKernelArg(conv_kernel, 6, sizeof(int), &in_w);
    err |= clSetKernelArg(conv_kernel, 7, sizeof(int), &kernel_h);
    err |= clSetKernelArg(conv_kernel, 8, sizeof(int), &kernel_w);
    err |= clSetKernelArg(conv_kernel, 9, sizeof(int), &out_h);
    err |= clSetKernelArg(conv_kernel, 10, sizeof(int), &out_w);
    err |= clSetKernelArg(conv_kernel, 11, sizeof(int), &do_relu);
    err |= clEnqueueNDRangeKernel(queue, conv_kernel, 3, NULL, global_conv3, NULL, 0, NULL, NULL);
    if(err != CL_SUCCESS) { printf("执行 conv3 出错\n"); return -1; }

    // 等待所有 kernel 执行完成
    clFinish(queue);

    // 7. 读取最终输出（conv3 输出为 10 个 float）
    float output[CONV3_OUT_CHANNELS];
    err = clEnqueueReadBuffer(queue, buf_conv3_out, CL_TRUE, 0,
                              sizeof(float)*CONV3_OUT_CHANNELS, output, 0, NULL, NULL);
    if(err != CL_SUCCESS) { printf("读取输出出错\n"); return -1; }

    // 8. 加载实际结果（测试数据的实际输出，由 Python 保存）
    int actual_count;
    float *actual_output = load_array_from_hex("./data_gen/test_output.txt", &actual_count);
    if(actual_count != CONV3_OUT_CHANNELS) {
        printf("实际结果数量(%d)与预期(%d)不符！\n", actual_count, CONV3_OUT_CHANNELS);
    }

    // 9. 输出对比结果
    printf("-------------------------------------------------\n");
    printf("索引\tOpenCL推理结果\t实际结果\t误差(%%)\tOpenCL结果(十六进制)\n");
    printf("-------------------------------------------------\n");
    for (int i = 0; i < CONV3_OUT_CHANNELS; i++) {
        char hex_buf[11]; // "h" + 8位数字 + '\0'
        float_to_hex_string(output[i], hex_buf);
        double err = (output[i] - actual_output[i]) / actual_output[i];
        printf("%d\t%f\t%f\t%.1f%\t%s\n", i, output[i], actual_output[i], err * 100.0, hex_buf);
    }
    printf("-------------------------------------------------\n");

    // 10. 释放资源
    clReleaseMemObject(buf_input);
    clReleaseMemObject(buf_conv1_out);
    clReleaseMemObject(buf_conv2_out);
    clReleaseMemObject(buf_conv3_out);
    clReleaseMemObject(buf_conv1_weight);
    clReleaseMemObject(buf_conv1_bias);
    clReleaseMemObject(buf_conv2_weight);
    clReleaseMemObject(buf_conv2_bias);
    clReleaseMemObject(buf_conv3_weight);
    clReleaseMemObject(buf_conv3_bias);
    clReleaseKernel(conv_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(conv1_weight);
    free(conv1_bias);
    free(conv2_weight);
    free(conv2_bias);
    free(conv3_weight);
    free(conv3_bias);
    free(test_input);
    free(actual_output);

    return 0;
}
