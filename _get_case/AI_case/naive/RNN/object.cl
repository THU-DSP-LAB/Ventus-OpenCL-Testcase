__kernel void RNN(__global float *a1, __global float *a2, __global float *b, __global float *c1, __global float *c2) {
  int tid = get_global_id(0);

    // 每个线程处理两个结果
// if (tid < 32) {
    // 计算对应的行和列索引 (处理 A1 和 C1)
    int row1 = tid;
    float sum1 = 0.0f;
    for (int k = 0; k < 8; ++k) {
        sum1 += a1[row1 * 8 + k] * b[k * 8 + tid % 8];
    }
    c1[row1 * 8 + tid % 8] += sum1;
    // if (c1[row1 * 8 + tid % 8] < 0.0f) {
    //     c1[row1 * 8 + tid % 8] = 0.0f;
    // }

    // 计算对应的行和列索引 (处理 A2 和 C2)
    int row2 = tid;
    float sum2 = 0.0f;
    for (int k = 0; k < 8; ++k) {
        sum2 += a2[row2 * 8 + k] * b[k * 8 + tid % 8];
    }
    c2[row2 * 8 + tid % 8] += sum2;
    // }
    // if (c2[row1 * 8 + tid % 8] < 0.0f) {
    //     c2[row1 * 8 + tid % 8] = 0.0f;
    // }


    // 每个线程处理两个结果
// if (tid < 32) {
    // 计算对应的行和列索引 (处理 A1 和 C1)
     row1 = tid;
     sum1 = 0.0f;
    for (int k = 0; k < 8; ++k) {
        sum1 += a1[row1 * 8 + k+32] * b[k * 8 + tid % 8+32];
    }
    c1[row1 * 8 + tid % 8] += sum1;
    if (c1[row1 * 8 + tid % 8] < 0.0f) {
        c1[row1 * 8 + tid % 8] = 0.0f;
    }

    // 计算对应的行和列索引 (处理 A2 和 C2)
     row2 = tid;
     sum2 = 0.0f;
    for (int k = 0; k < 8; ++k) {
        sum2 += a2[row2 * 8 + k+32] * b[k * 8 + tid % 8+32];
    }
    c2[row2 * 8 + tid % 8] += sum2;
    // }
    if (c2[row1 * 8 + tid % 8] < 0.0f) {
        c2[row1 * 8 + tid % 8] = 0.0f;
    }

}