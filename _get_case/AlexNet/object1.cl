// relu.cl — y = max(0, x)
__kernel void relu3d(
    __global const float* x,   // [C,H,W]
    __global float*       y,   // [C,H,W]
    const int C,
    const int H,
    const int W
){
    const int c = get_global_id(0);
    const int yy= get_global_id(1);
    const int xx= get_global_id(2);
    if (c >= C || yy >= H || xx >= W) return;

    const int idx = c * (H * W) + yy * W + xx;
    float v = x[idx];
    y[idx] = (v > 0.0f) ? v : 0.0f;
}
