// batchnorm.cl

__kernel void batchnorm_inference(
    __global const float* input,      // NCHW
    __global const float* running_mean,  // [C]
    __global const float* running_var,   // [C]
    __global const float* gamma,         // [C] (scale)
    __global const float* beta,          // [C] (bias)
    __global float* output,              // NCHW
    const int N, const int C, const int H, const int W,
    const float eps
) {
    int gid = get_global_id(0);
    int total = N * C * H * W;
    if (gid >= total) return;

    int n = gid / (C * H * W);
    int c = (gid / (H * W)) % C;
    int h = (gid / W) % H;
    int w = gid % W;

    int index = ((n * C + c) * H + h) * W + w;

    float mean = running_mean[c];
    float var = running_var[c];
    float scale = gamma[c];
    float shift = beta[c];

    float norm = (input[index] - mean) / sqrt(var + eps);
    output[index] = norm * scale + shift;
}
