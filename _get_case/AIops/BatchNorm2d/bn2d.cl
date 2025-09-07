// bn2d.cl — BatchNorm2d (inference): y = gamma * (x - mean) / sqrt(var + eps) + beta
__kernel void batchnorm2d_infer(
    __global const float* x,      // [C,H,W]
    __global const float* gamma,  // [C]
    __global const float* beta,   // [C]
    __global const float* mean,   // [C]
    __global const float* var,    // [C]
    const float eps,
    __global float* y,            // [C,H,W]
    const int C,
    const int H,
    const int W
){
    const int c = get_global_id(0);
    const int y0= get_global_id(1);
    const int x0= get_global_id(2);
    if (c >= C || y0 >= H || x0 >= W) return;

    const int idx = c * (H * W) + y0 * W + x0;
    const float inv_std = 1.0f / sqrt(var[c] + eps);
    y[idx] = gamma[c] * (x[idx] - mean[c]) * inv_std + beta[c];
}
