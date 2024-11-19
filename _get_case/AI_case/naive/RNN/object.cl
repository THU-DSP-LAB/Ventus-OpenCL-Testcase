__kernel void RNN(__global float *a1, __global float *a2, __global float *b, __global float *c1, __global float *c2) {
  int tid = get_global_id(0);

    // 每个线程处理两个结果
// _____________________________第1次矩阵乘法————————————————————————————————
    // 计算对应的行和列索引 (处理 A1 和 C1)，
    // 完成32*2=64元素的C矩阵输出
        // 处理计算32个FP32结果 对应C 8*4
    int row1 = tid/8;
    float sum1 = 0.0f;
    for (int k = 0; k < 8; ++k) {
        sum1 += a1[row1 * 8 + k] * b[k * 8 + tid % 8];
    }
    c1[row1 * 8 + tid % 8] += sum1;
        // 处理计算32个FP32结果 对应C 8*4
    row1 = tid/8 + 4;
    sum1 = 0.0f;
    for (int k = 0; k < 8; ++k) {
        sum1 += a1[row1 * 8 + k] * b[k * 8 + tid % 8];
    }
    c1[tid+32] += sum1;

    // 计算对应的行和列索引 (处理 A2 和 C2) 
    // 完成32*2=64元素的C矩阵输出
        // 处理计算32个FP32结果 对应C 8*4
    int row2 = tid/8;
    float sum2 = 0.0f;
    for (int k = 0; k < 8; ++k) {
        sum2 += a2[row2 * 8 + k] * b[k * 8 + tid % 8];
    }
    c2[row2 * 8 + tid % 8] += sum2;
        // 处理计算32个FP32结果 对应C 8*4
    row2 = tid/8+4;
    sum2 = 0.0f;
    for (int k = 0; k < 8; ++k) {
        sum2 += a2[row2 * 8 + k] * b[k * 8 + tid % 8];
    }
    c2[tid+32] += sum2;

// _____________________________第二次累加——————c不变 ab偏移——————————————————————————
    // 每个线程处理两个结果
    // 计算对应的行和列索引 (处理 A1 和 C1)
    row1 = tid/8;
    sum1 = 0.0f;
    for (int k = 0; k < 8; ++k) {
        sum1 += a1[row1 * 8 + k+64] * b[k * 8 + tid % 8+64];
    }
    c1[tid] += sum1;
    if (c1[tid] < 0.0f) {
        c1[tid] = 0.0f;
    }
    row1 = tid/8+4;
    sum1 = 0.0f;
    for (int k = 0; k < 8; ++k) {
        sum1 += a1[row1 * 8 + k+64] * b[k * 8 + tid % 8+64];
    }
    c1[tid+32] += sum1;
    if (c1[tid+32] < 0.0f) {
        c1[tid+32] = 0.0f;
    }

    // 计算对应的行和列索引 (处理 A2 和 C2)
    row2 = tid/8;
    sum2 = 0.0f;
    for (int k = 0; k < 8; ++k) {
        sum2 += a2[row2 * 8 + k+64] * b[k * 8 + tid % 8+64];
    }
    c2[tid] += sum2;
    // }
    if (c2[tid] < 0.0f) {
        c2[tid] = 0.0f;
    }

    row2 = tid/8+4;
    sum2 = 0.0f;
    for (int k = 0; k < 8; ++k) {
        sum2 += a2[row2 * 8 + k+64+32] * b[k * 8 + tid % 8+96];
    }
    c2[tid+32] += sum2;
    // }
    if (c2[tid+32] < 0.0f) {
        c2[tid+32] = 0.0f;
    }

}