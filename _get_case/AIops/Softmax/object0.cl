// softmax.cl — 使用内建 exp() 的数值稳定 softmax

/* 1) 行 softmax：对每一行长度 N 的向量做 softmax
   输入/输出：x,y 形状 [M, N]（行主序），y 可与 x 同/不同缓冲区
*/
__kernel void softmax_rows(
    __global const float* x,   // [M*N]
    __global float*       y,   // [M*N]
    const int M,
    const int N
){
    const int r = get_global_id(0);   // 行索引
    if (r >= M) return;

    const int base = r * N;

    // 1) max
    float m = x[base];
    for (int j=1; j<N; ++j){
        float v = x[base + j];
        m = v > m ? v : m;
    }

    // 2) sum exp(x - m), 暂存到 y
    float sum = 0.0f;
    for (int j=0; j<N; ++j){
        float e = exp(x[base + j] - m);
        y[base + j] = e;
        sum += e;
    }

    // 3) 归一化
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int j=0; j<N; ++j){
        y[base + j] *= inv_sum;
    }
}

/* 2) NCHW（N=1）下，对 C 做 softmax：每个 (y,x) 位置独立
   输入/输出：x,y 形状 [C, H, W]，索引 idx = c*(H*W) + y*W + x
*/
__kernel void softmax_chw(
    __global const float* x,   // [C,H,W]
    __global float*       y,   // [C,H,W]
    const int C,
    const int H,
    const int W
){
    const int yy = get_global_id(0);
    const int xx = get_global_id(1);
    if (yy >= H || xx >= W) return;

    const int HW = H * W;
    const int col = yy * W + xx;

    // 1) max over c
    float m = x[col];
    for (int c=1; c<C; ++c){
        float v = x[c*HW + col];
        m = v > m ? v : m;
    }

    // 2) sum exp(x - m), 暂存到 y
    float sum = 0.0f;
    for (int c=0; c<C; ++c){
        float e = exp(x[c*HW + col] - m);
        y[c*HW + col] = e;
        sum += e;
    }

    // 3) 归一化
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int c=0; c<C; ++c){
        y[c*HW + col] *= inv_sum;
    }
}
