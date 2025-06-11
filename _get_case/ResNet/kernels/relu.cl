// relu.cl

__kernel void relu(
    __global const float* input,
    __global float* output,
    const int total_size
) {
    int gid = get_global_id(0);
    if (gid < total_size) {
        output[gid] = fmax(input[gid], 0.0f);
    }
}
