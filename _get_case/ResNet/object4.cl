__kernel void gemm(
    __global const float* A,   // A 矩阵 (M x K)
    __global const float* B,   // B 矩阵 (K x N)
    __global float* C,         // 输出矩阵 C (M x N)
    const int M,               // A 矩阵行数
    const int K,               // A 矩阵列数与 B 矩阵行数
    const int N,               // B 矩阵列数
    const int do_trans_a,      // 是否转置 A 矩阵
    const int do_trans_b       // 是否转置 B 矩阵
)
{
    int row = get_global_id(0);  // 输出矩阵行
    int col = get_global_id(1);  // 输出矩阵列

    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        float a_val = (do_trans_a) ? A[k * M + row] : A[row * K + k];
        float b_val = (do_trans_b) ? B[col * K + k] : B[k * N + col];

        sum += a_val * b_val;
    }

    C[row * N + col] = sum;
}
