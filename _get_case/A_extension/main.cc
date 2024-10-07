#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <chrono>

#define ALIGNMENT 4
// 定义一个共享的全局变量，并确保其对齐
volatile uint32_t shared_data __attribute__((aligned(4))) = 0;
// 检查OpenCL调用的宏
#define CL_CHECK(err) if(err != CL_SUCCESS) { printf("OpenCL Error: %d\n", err); exit(-1); }

// 读取内核文件
int read_kernel_file(const char* filename, char** kernel_bin, size_t* kernel_size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("Failed to open kernel file.\n");
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    *kernel_size = ftell(fp);
    rewind(fp);

    *kernel_bin = (char*)malloc(*kernel_size);
    fread(*kernel_bin, 1, *kernel_size, fp);
    fclose(fp);

    return 0;
}

int main(int argc, char** argv) {
    // 初始化OpenCL平台和设备
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem A_buffer, B_buffer, C_buffer;
    size_t kernel_size;
    char* kernel_bin;

    CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
    CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device_id, 0, NULL);

    // 读取内核文件
    if (read_kernel_file("kernel.cl", &kernel_bin, &kernel_size) != 0) {
        return -1;
    }

    // 创建程序对象
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_bin, &kernel_size, NULL);
    free(kernel_bin);

    // 编译程序
    CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

    // 创建内核对象
    kernel = clCreateKernel(program, "A_extension", NULL);

    // 创建缓冲区
    int size = 32;//32个thread/work-item
    int datasize = size * 10;
    size_t nbytes = sizeof(float) * datasize;
    A_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, NULL);
    B_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, NULL);
    C_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, NULL);

    // 设置内核参数
    clSetKernelArg(kernel, 0, sizeof(A_buffer), &A_buffer);
    clSetKernelArg(kernel, 1, sizeof(B_buffer), &B_buffer);
    clSetKernelArg(kernel, 2, sizeof(C_buffer), &C_buffer);

    // clSetKernelArg(kernel, 2, sizeof(int), &size);
    // 分配局部内存
    // size_t local_mem_size = sizeof(float) * size;
    // clSetKernelArg(kernel, 2, local_mem_size, NULL);

    // 初始化输入数据
    // int size = size * 256;
    std::vector<int> input(datasize);

    size_t alignment = 4;  // 4字节对齐
    int *data = (int *)aligned_alloc(alignment, datasize);
    if (data == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(-1);
    }
    // // int *input = nullptr;
    
    // // 使用 while 循环确保内存对齐
    // int *data = nullptr;
    // void *raw_data = nullptr;
    // while (true) {
    //     raw_data = malloc(sizeof(int) * datasize + ALIGNMENT);
    //     if (raw_data == nullptr) {
    //         fprintf(stderr, "Memory allocation failed, retrying...\n");
    //         continue;
    //     }

    //     // 手动对齐
    //     data = (int *)(((uintptr_t)raw_data + ALIGNMENT - 1) & ~(ALIGNMENT - 1));

    //     // 检查对齐
    //     if (((uintptr_t)data & (ALIGNMENT - 1)) == 0) {
    //         break;
    //     } else {
    //         fprintf(stderr, "Memory not aligned, retrying...\n");
    //         free(raw_data);
    //     }
    // }
    
    // 初始化input为0~100的随机浮点数
    for(int i = 0; i < datasize; i++){
      input[i] = static_cast<int>(rand() % 101);
    }
    printf("\n");

    // 将输入数据拷贝到设备端
    CL_CHECK(clEnqueueWriteBuffer(queue, A_buffer, CL_TRUE, 0, nbytes, input.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(queue, B_buffer, CL_TRUE, 0, nbytes, input.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(queue, C_buffer, CL_TRUE, 0, nbytes, input.data(), 0, NULL, NULL));


    // 设置工作项和工作组的大小
    size_t global_work_size[1] = { static_cast<size_t>(size) };
    size_t local_work_size[1] = { static_cast<size_t>(size) };

    // 执行内核
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    CL_CHECK(clFinish(queue));

    // 读取输出数据
    std::vector<float> output(datasize);
    CL_CHECK(clEnqueueReadBuffer(queue, C_buffer, CL_TRUE, 0, nbytes, output.data(), 0, NULL, NULL));

    // 验证结果
    // for (int i = 0; i < datasize; ++i) {
    //     if (output[i] != input[i] * 2.0f) {
    //         printf("Error at index %d: expected %f, got %f\n", i, input[i] * 2.0f, output[i]);
    //     }
    // }
    // printf("finished kernel and success if no error!\n");

    // 清理资源
    clReleaseMemObject(A_buffer);
    clReleaseMemObject(B_buffer);
    clReleaseMemObject(C_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    printf("Done.\n");
    return 0;
}
