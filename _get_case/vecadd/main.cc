#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <netinet/in.h>
#include <iostream>


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

// Function to remove the 'h' prefix from the string
std::string removeHPrefix(const std::string& str) {
    if (str.size() >= 1 && str[0] == 'h') {
        return str.substr(1);
    }
    return str;
}

// Function to convert an octal string to a uint32_t
uint32_t octalStringToUint32(const std::string& octalStr) {
    std::string trimmedStr = removeHPrefix(octalStr);

    // std::cout << trimmedStr<<" - ";

    std::stringstream ss;
    uint32_t value;
    ss << std::hex << trimmedStr;
    ss >> value;
    // std::cout << value<<" * ";
    return value;
}

// Function to read the file and parse the octal strings into uint32_t
std::vector<uint32_t> parseOctalFile(const std::string& filePath, size_t size) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }

    std::string line;
    std::getline(file, line);

    std::istringstream iss(line);
    std::vector<uint32_t> values;

    std::string token;
    while (iss >> token) {
        values.push_back(octalStringToUint32(token));
    }

    std::vector<uint32_t> padded(values.begin(), values.end());
    padded.resize(size, 0);  // Resize the vector and fill with zeros if necessary
    
    return padded;
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
    if (read_kernel_file("vecadd.cl", &kernel_bin, &kernel_size) != 0) {
        return -1;
    }

    // 创建程序对象
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_bin, &kernel_size, NULL);
    free(kernel_bin);

    // 编译程序
    CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

    // 创建内核对象
    kernel = clCreateKernel(program, "vecadd", NULL);

    // 创建缓冲区
    int size = 32;//32个thread/work-item
    int datasize = size * 10;
    size_t nbytes = sizeof(float) * datasize;
    A_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, NULL);
    B_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, NULL);
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
    // std::vector<float> input(datasize);
    // // 初始化input为0~100的随机浮点数
    // for(int i = 0; i < datasize; i++){
    //   input[i] = static_cast<float>(rand() % 101);
    // }
    // printf("\n");

    std::string filePath = "./testData_888/RA.txt";
    std::vector<uint32_t> inA = parseOctalFile(filePath,datasize);
    filePath = "./testData_888/RB.txt";
    std::vector<uint32_t> inB = parseOctalFile(filePath,datasize);
    filePath = "./testData_888/RC.txt";
    std::vector<uint32_t> inC = parseOctalFile(filePath,datasize); 
    for (const auto& value : inC) {
        std::cout << std::hex << value << " ";
    }

    // 将输入数据拷贝到设备端
    CL_CHECK(clEnqueueWriteBuffer(queue, A_buffer, CL_TRUE, 0, nbytes, inA.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(queue, B_buffer, CL_TRUE, 0, nbytes, inB.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(queue, C_buffer, CL_TRUE, 0, nbytes, inC.data(), 0, NULL, NULL));


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
