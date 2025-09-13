// add.cl — Elementwise add: out = a + b
__kernel void add3d(
    __global const float* a,   // [C,H,W]
    __global const float* b,   // [C,H,W]
    __global float*       out, // [C,H,W]
    const int C,
    const int H,
    const int W
){
    const int c = get_global_id(0);
    const int y = get_global_id(1);
    const int x = get_global_id(2);
    if (c >= C || y >= H || x >= W) return;

    const int idx = c * (H * W) + y * W + x;
    out[idx] = a[idx] + b[idx];
}
