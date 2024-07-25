// #include <stdio.h>
// #include <stdlib.h>
// #include "poclu.h"

// #define ARRAY_SIZE 32

// // 错误检查宏定义
// #define CHECK_POCLU_ERROR(err, msg) if (err != CL_SUCCESS) { \
//     printf("PoCL Error: %s (%d)\n", msg, err); \
//     exit(1); \
// }

// int main() {
//     // 定义和初始化数组A、B、C为float类型
//     float A[ARRAY_SIZE] = {0};
//     float B[ARRAY_SIZE] = {0};
//     float C[ARRAY_SIZE] = {0};
//     for (int i = 0; i < ARRAY_SIZE; i++) {
//         A[i] = 1.0f; // 赋值为1.0
//         B[i] = 1.0f;
//     }

//     // PoCL初始化
//     cl_device_id device;
//     cl_context context;
//     cl_command_queue queue;
//     int err = poclu_get_any_device(&context, &device);
//     CHECK_POCLU_ERROR(err, "Unable to get PoCL device");

//     queue = clCreateCommandQueue(context, device, 0, &err);
//     CHECK_POCLU_ERROR(err, "Unable to create command queue");

//     // 内核代码字符串
//     const char *kernel_code =
//         "__kernel void mma(__global const float *a, __global const float *b, __global float *c) {\n"
//         "  int tid = get_global_id(0);\n"
//         "  __asm__ volatile(\".insn r 0x7b, 2, 1, %0, %1, %2\" : \"+r\" (c[tid]) : \"r\" (a[tid]), \"r\" (b[tid]));\n"
//         "}";

//     // 创建并构建程序
//     cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, &err);
//     CHECK_POCLU_ERROR(err, "Unable to create program with source");

//     err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
//     CHECK_POCLU_ERROR(err, "Unable to build program");

//     // 创建内核
//     cl_kernel kernel = clCreateKernel(program, "mma", &err);
//     CHECK_POCLU_ERROR(err, "Unable to create kernel");

//     // 设置内核参数
//     err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)A);
//     err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)B);
//     err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)C);
//     CHECK_POCLU_ERROR(err, "Unable to set kernel arguments");

//     // 执行内核
//     size_t global_size = ARRAY_SIZE;
//     err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
//     CHECK_POCLU_ERROR(err, "Unable to enqueue kernel");

//     // 读取结果
//     err = clEnqueueReadBuffer(queue, clCreateBuffer(context, CL_MEM_USE_HOST_PTR, ARRAY_SIZE * sizeof(float), C, NULL), CL_TRUE, 0, ARRAY_SIZE * sizeof(float), C, 0, NULL, NULL);
//     CHECK_POCLU_ERROR(err, "Unable to read buffer");

//     // 清理资源
//     clReleaseKernel(kernel);
//     clReleaseProgram(program);
//     clReleaseCommandQueue(queue);
//     clReleaseContext(context);

//     // 打印结果
//     for (int i = 0; i < ARRAY_SIZE; i++) {
//         printf("C[%d] = %f\n", i, C[i]);
//     }

//     return 0;
// }