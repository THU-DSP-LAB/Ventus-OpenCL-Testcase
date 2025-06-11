#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cstring>   // for memcpy
#include <cstdint>   // for uint32_t

// 检查 OpenCL 调用错误的宏
#define CL_CHECK(err) if(err != CL_SUCCESS){ std::cerr << "OpenCL Error: " << err << std::endl; exit(-1); }

// 读取内核文件，返回内核源代码字符串和其长度
int read_kernel_file(const char* filename, char** kernel_source, size_t* kernel_size) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()){
        std::cerr << "Failed to open kernel file: " << filename << std::endl;
        return -1;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    *kernel_source = new char[size + 1];
    if (!file.read(*kernel_source, size)){
        std::cerr << "Failed to read kernel file." << std::endl;
        return -1;
    }
    (*kernel_source)[size] = '\0';
    *kernel_size = static_cast<size_t>(size);
    return 0;
}

// 将形如 "h3f800000" 的 hex 字符串转换为 float
float hexStringToFloat(const std::string &hexStr) {
    std::string token = hexStr;
    if (!token.empty() && (token[0] == 'h' || token[0] == 'H'))
        token = token.substr(1);
    uint32_t intVal = 0;
    std::stringstream ss;
    ss << std::hex << token;
    ss >> intVal;
    float f;
    std::memcpy(&f, &intVal, sizeof(float));
    return f;
}

// 从指定文件中读取一行数据，按空格分隔，将 hex 字符串转换为 float，并返回 vector<float>
// 若实际数据少于 expected_count，则用0补足；若多于 expected_count，则截断
std::vector<float> readHexFile(const std::string &filename, int expected_count) {
    std::ifstream file(filename);
    if(!file.is_open()){
         std::cerr << "Failed to open file: " << filename << std::endl;
         exit(-1);
    }
    std::string line;
    std::getline(file, line);
    file.close();

    std::istringstream iss(line);
    std::vector<float> data;
    std::string token;
    while(iss >> token) {
         data.push_back(hexStringToFloat(token));
    }
    if (data.size() < static_cast<size_t>(expected_count)) {
         data.resize(expected_count, 0.0f);
    } else if (data.size() > static_cast<size_t>(expected_count)) {
         data.resize(expected_count);
    }
    return data;
}

// 将浮点数转换为16进制字符串
std::string floatToHex(float f) {
    uint32_t intVal;
    std::memcpy(&intVal, &f, sizeof(uint32_t));
    std::stringstream ss;
    ss << std::hex << intVal;
    return "h" + ss.str();
}

int main() {
    cl_int err;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // 1. 获取平台和设备（此处选择默认设备）
    CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
    CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

    // 2. 创建上下文和命令队列
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    CL_CHECK(err);
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    CL_CHECK(err);

    // 3. 读取内核文件（此处文件名为 nn_forward.cl）
    char* kernel_source = nullptr;
    size_t kernel_size;
    if (read_kernel_file("nn_forward.cl", &kernel_source, &kernel_size) != 0) {
        return -1;
    }

    // 4. 创建程序对象并编译
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_size, &err);
    CL_CHECK(err);
    delete[] kernel_source;  // 释放读取的内核字符串
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // 输出编译日志
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), NULL);
        std::cerr << "Build log:\n" << build_log.data() << std::endl;
        exit(1);
    }

    // 5. 创建内核对象，内核名为 "fc_layer"
    kernel = clCreateKernel(program, "fc_layer", &err);
    CL_CHECK(err);

    // 6. 定义神经网络尺寸（MNIST 数据：输入 784、隐藏 128、输出 10）
    const int INPUT_SIZE  = 784;
    const int HIDDEN_SIZE = 128;
    const int OUTPUT_SIZE = 10;

    // 7. 从文件中加载各层权重、偏置和测试数据
    std::vector<float> input = readHexFile("./data_gen/test_input.txt", INPUT_SIZE);
    std::vector<float> weight1 = readHexFile("./data_gen/fc1_weight.txt", INPUT_SIZE * HIDDEN_SIZE);
    std::vector<float> bias1   = readHexFile("./data_gen/fc1_bias.txt", HIDDEN_SIZE);
    std::vector<float> weight2 = readHexFile("./data_gen/fc2_weight.txt", HIDDEN_SIZE * OUTPUT_SIZE);
    std::vector<float> bias2   = readHexFile("./data_gen/fc2_bias.txt", OUTPUT_SIZE);
    // 额外加载期望输出（用于对比）
    std::vector<float> expected_output = readHexFile("./data_gen/test_output.txt", OUTPUT_SIZE);
    
    // 用于存储前向计算结果
    std::vector<float> output(OUTPUT_SIZE, 0.0f);

    // 8. 创建设备端缓冲区
    cl_mem input_buf   = clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, sizeof(float) * INPUT_SIZE, input.data(), &err);
    CL_CHECK(err);
    cl_mem weight1_buf = clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, sizeof(float) * INPUT_SIZE * HIDDEN_SIZE, weight1.data(), &err);
    CL_CHECK(err);
    cl_mem bias1_buf   = clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, sizeof(float) * HIDDEN_SIZE, bias1.data(), &err);
    CL_CHECK(err);
    cl_mem hidden_buf  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * HIDDEN_SIZE, NULL, &err);
    CL_CHECK(err);
    cl_mem weight2_buf = clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, sizeof(float) * HIDDEN_SIZE * OUTPUT_SIZE, weight2.data(), &err);
    CL_CHECK(err);
    cl_mem bias2_buf   = clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, sizeof(float) * OUTPUT_SIZE, bias2.data(), &err);
    CL_CHECK(err);
    cl_mem output_buf  = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * OUTPUT_SIZE, NULL, &err);
    CL_CHECK(err);

    // 9. 根据硬件资源限制（每 SM 4 warp，每 warp 8 线程，共 2 SM）设置 NDRange 参数
    // 第一层：输入→隐藏层（HIDDEN_SIZE 个神经元）
    size_t local_work_size1 = 32; // 每组 32 个线程
    size_t global_work_size1 = ((HIDDEN_SIZE + local_work_size1 - 1) / local_work_size1) * local_work_size1;

    // 设置 fc_layer 内核参数（第一层：输入→隐藏，激活置为 1，即使用 ReLU）
    err  = clSetKernelArg(kernel, 0, sizeof(int), &INPUT_SIZE);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &HIDDEN_SIZE);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &input_buf);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &weight1_buf);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &bias1_buf);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &hidden_buf);
    int activation_relu = 1;
    err |= clSetKernelArg(kernel, 6, sizeof(int), &activation_relu);
    CL_CHECK(err);

    // 执行第一层内核
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size1, &local_work_size1, 0, NULL, NULL);
    CL_CHECK(err);

    // 为了调试，将第一层结果从 hidden_buf 读回 host，并输出到 log
    std::vector<float> hidden_result(HIDDEN_SIZE, 0.0f);
    err = clEnqueueReadBuffer(queue, hidden_buf, CL_TRUE, 0, sizeof(float) * HIDDEN_SIZE, hidden_result.data(), 0, NULL, NULL);
    CL_CHECK(err);
    std::cout << "First layer (hidden) output:" << std::endl;
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        std::cout << "Hidden[" << i << "] = " << hidden_result[i]
                  << " in hex: " << floatToHex(hidden_result[i]) << std::endl;
    }

    // 第二层：隐藏→输出层（OUTPUT_SIZE 个神经元）
    size_t local_work_size2 = 32;
    size_t global_work_size2 = ((OUTPUT_SIZE + local_work_size2 - 1) / local_work_size2) * local_work_size2;

    // 设置 fc_layer 内核参数（第二层：隐藏→输出，激活置为 0，不使用激活）
    err  = clSetKernelArg(kernel, 0, sizeof(int), &HIDDEN_SIZE);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &OUTPUT_SIZE);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &hidden_buf);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &weight2_buf);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &bias2_buf);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &output_buf);
    int activation_none = 0;
    err |= clSetKernelArg(kernel, 6, sizeof(int), &activation_none);
    CL_CHECK(err);

    // 执行第二层内核
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size2, &local_work_size2, 0, NULL, NULL);
    CL_CHECK(err);

    // 等待所有命令执行完毕
    clFinish(queue);

    // 10. 读取设备端输出层结果
    err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, sizeof(float) * OUTPUT_SIZE, output.data(), 0, NULL, NULL);
    CL_CHECK(err);

    // 输出前向计算结果及期望输出（来自 test_output.txt）
    std::cout << "Neural network forward output:" << std::endl;
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        std::cout << "Output[" << i << "] = " << output[i]
                  << " (expected " << expected_output[i] << ")"
                  << " in hex: " << floatToHex(output[i]) << std::endl;
    }

    // 11. 释放所有 OpenCL 资源
    clReleaseMemObject(input_buf);
    clReleaseMemObject(weight1_buf);
    clReleaseMemObject(bias1_buf);
    clReleaseMemObject(hidden_buf);
    clReleaseMemObject(weight2_buf);
    clReleaseMemObject(bias2_buf);
    clReleaseMemObject(output_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
