// matmul.cl — 单 work-group 的通用 SGEMM（C = A[MxK] * B[KxN]）
// 约束：只允许 1 个 work-group。通过 host 设置 global==local 实现。
// 使用 tile（TILE_M x TILE_N）并在 K 维上做分块 TILE_K。
// TILE_* 通过 build 选项 -DTILE_M=.. -DTILE_N=.. -DTILE_K=.. 传入（host 已设置）。
#ifndef TILE_M
#define TILE_M 16
#endif
#ifndef TILE_N
#define TILE_N 16
#endif
#ifndef TILE_K
#define TILE_K 16
#endif

__kernel void sgemm_one_wg(__global const float* A,
                           __global const float* B,
                           __global float*       C,
                           const int M, const int N, const int K)
{
    // 只一个 work-group，所有 work-items 协同覆盖整张矩阵
    const int lx = get_local_id(0); // tile 内列
    const int ly = get_local_id(1); // tile 内行

    __local float As[TILE_M*TILE_K];
    __local float Bs[TILE_K*TILE_N];

    // 遍历输出矩阵的所有 tile（行块、列块）
    for (int bm = 0; bm < M; bm += TILE_M) {
        for (int bn = 0; bn < N; bn += TILE_N) {

            // 该线程负责的输出元素坐标
            const int row = bm + ly;
            const int col = bn + lx;

            float acc = 0.0f;

            // K 方向分块
            for (int bk = 0; bk < K; bk += TILE_K) {
                // 装载 A 的一个子块到本地内存
                int a_row = row;
                int a_col = bk + lx; // 每列由 lx 分担
                As[ly*TILE_K + lx] = (a_row < M && a_col < K) ? A[a_row*K + a_col] : 0.0f;

                // 装载 B 的一个子块到本地内存
                int b_row = bk + ly; // 每行由 ly 分担
                int b_col = col;
                Bs[ly*TILE_N + lx] = (b_row < K && b_col < N) ? B[b_row*N + b_col] : 0.0f;

                barrier(CLK_LOCAL_MEM_FENCE);

                // 累加该 K 子块
                for (int kk = 0; kk < TILE_K; ++kk) {
                    float a = As[ly*TILE_K + kk];
                    float b = Bs[kk*TILE_N + lx];
                    acc += a * b;
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (row < M && col < N) {
                C[row*N + col] = acc;
            }
            // 下一对 (bm,bn) 由同一批线程接力计算，直到覆盖完整 C
        }
    }
}
