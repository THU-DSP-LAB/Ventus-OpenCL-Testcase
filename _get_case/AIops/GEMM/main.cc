#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* ── 网络超参 ───────────────────────── */
const int M = 512;   // A 矩阵行数
const int N = 512;   // B 矩阵列数
const int K = 64;   // A 矩阵列数与 B 矩阵行数
const int DO_TRANS_A = 0;  // 是否对A矩阵进行转置
const int DO_TRANS_B = 0;  // 是否对B矩阵进行转置

/* ── 工具函数 ───────────────────────── */
void generate_random_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)(rand() % 1000) / 100.0f;  // 生成 [0.0, 10.0) 之间的随机数
    }
}

// 加载 kernel 源代码
char* load_kernel_source(const char* file) {
    FILE* fp = fopen(file, "r");
    if (!fp) {
        fprintf(stderr, "无法打开 %s\n", file);
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    rewind(fp);
    char* src = (char*)malloc(sz + 1);
    fread(src, 1, sz, fp);
    src[sz] = '\0';
    fclose(fp);
    return src;
}

// 参考 GEMM 实现 (CPU 端)
void gemm_reference(float* A, float* B, float* C, int M, int K, int N, int transA, int transB) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_val = (transA) ? A[k * M + i] : A[i * K + k];
                float b_val = (transB) ? B[j * K + k] : B[k * N + j];
                sum += a_val * b_val;
            }
            C[i * N + j] = sum;
        }
    }
}

/* ── main ──────────────────────────── */
int main(void) {
    srand(time(NULL));  // 初始化随机种子

    cl_int err;

    /* 1) 平台 / 设备 */
    cl_platform_id plat;
    err = clGetPlatformIDs(1, &plat, NULL);
    cl_device_id dev;
    err |= clGetDeviceIDs(plat, CL_DEVICE_TYPE_DEFAULT, 1, &dev, NULL);

    /* 2) 上下文 / 队列 */
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);

    /* 3) program / kernel */
    char* src = load_kernel_source("gemm.cl");
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, NULL, &err);
    free(src);
    err |= clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
    cl_kernel k = clCreateKernel(prog, "gemm", &err);

    /* 4) 随机生成 A 和 B 矩阵 */
    int a_size = M * K;
    int b_size = K * N;
    float *A = (float*)malloc(a_size * sizeof(float));
    float *B = (float*)malloc(b_size * sizeof(float));
    float *C = (float*)malloc(M * N * sizeof(float));  // OpenCL 计算结果
    float *C_ref = (float*)malloc(M * N * sizeof(float));  // 参考计算结果

    generate_random_matrix(A, M, K);
    generate_random_matrix(B, K, N);

    /* 5) 创建缓冲区 */
    cl_mem buf_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * a_size, A, &err);
    cl_mem buf_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * b_size, B, &err);
    cl_mem buf_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * M * N, NULL, &err);

    /* 6) 创建变量 */
    int m_val = M;
    int k_val = K;
    int n_val = N;
    int trans_a_val = DO_TRANS_A;
    int trans_b_val = DO_TRANS_B;

    /* 7) 设置 kernel 参数 */
    err = clSetKernelArg(k, 0, sizeof(cl_mem), &buf_A);
    err |= clSetKernelArg(k, 1, sizeof(cl_mem), &buf_B);
    err |= clSetKernelArg(k, 2, sizeof(cl_mem), &buf_C);
    err |= clSetKernelArg(k, 3, sizeof(int), &m_val);
    err |= clSetKernelArg(k, 4, sizeof(int), &k_val);
    err |= clSetKernelArg(k, 5, sizeof(int), &n_val);
    err |= clSetKernelArg(k, 6, sizeof(int), &trans_a_val);
    err |= clSetKernelArg(k, 7, sizeof(int), &trans_b_val);

    /* 8) 设置工作组 */
    size_t global_work_size[2] = {M, N};
    err |= clEnqueueNDRangeKernel(q, k, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

    /* 9) 获取 OpenCL 结果 */
    err |= clEnqueueReadBuffer(q, buf_C, CL_TRUE, 0, sizeof(float) * M * N, C, 0, NULL, NULL);
    clFinish(q);

    /* 10) 参考计算 */
    gemm_reference(A, B, C_ref, M, K, N, DO_TRANS_A, DO_TRANS_B);

    /* 11) 比较结果 */
    int error_count = 0;
    for (int i = 0; i < M * N; i++) {
        if (fabs(C[i] - C_ref[i]) > 1e-3) {
            error_count++;
            if (i < 10) {
                printf("Error at index %d: OpenCL = %f, Reference = %f\n", i, C[i], C_ref[i]);
            }
        }
    }
    printf("\nTotal errors: %d / %d\n", error_count, M * N);

    /* 12) 输出结果并释放资源 */
    free(A);
    free(B);
    free(C);
    free(C_ref);

    clReleaseMemObject(buf_A);
    clReleaseMemObject(buf_B);
    clReleaseMemObject(buf_C);
    clReleaseKernel(k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    return 0;
}
