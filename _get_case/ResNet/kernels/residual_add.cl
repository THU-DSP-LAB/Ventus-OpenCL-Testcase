// residual_add.cl

__kernel void residual_add(
    __global const float* x,
    __global const float* y,
    __global float* out,
    const int total_size
) {
    int gid = get_global_id(0);
    if (gid < total_size) {
        out[gid] = x[gid] + y[gid];
    }
}
