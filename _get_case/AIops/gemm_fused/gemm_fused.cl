__kernel void gemm_relu(
    __global const float* A,        // 输入矩阵 A
    __global const float* B,        // 输入矩阵 B
    __global float* C,              // 输出矩阵 C
    const int M, const int K, const int N, // A的行数、列数，B的行数和列数
    const int do_trans_a, const int do_trans_b // 是否转置矩阵 A 或 B
) {
    int m = get_global_id(0);  // A的行索引
    int n = get_global_id(1);  // B的列索引

    if (m >= M || n >= N) return;

    float value = 0.0f;

    // 矩阵乘法
    for (int k = 0; k < K; ++k) {
        int a_idx = (do_trans_a ? k : m) * K + (do_trans_a ? m : k);  // 根据是否转置，选择正确的索引
        int b_idx = (do_trans_b ? n : k) * N + (do_trans_b ? k : n);

        value += A[a_idx] * B[b_idx];
    }

    // ReLU 激活
    C[m * N + n] = max(0.0f, value);
}
