// conv2d.cl

__kernel void conv2d_nchw(
    __global const float* input,      // NCHW
    __global const float* weight,     // OIHW
    __global const float* bias,       // optional, size = out_channels
    __global float* output,           // NCHW
    const int N, const int C, const int H, const int W,
    const int out_channels,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w
) {
    int n = get_global_id(0);      // batch index
    int oc = get_global_id(1);     // output channel
    int oh = get_global_id(2);     // output height
    int ow = get_global_id(3);     // output width

    int out_h = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    float acc = bias ? bias[oc] : 0.0f;

    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = oh * stride_h - pad_h + kh;
                int iw = ow * stride_w - pad_w + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int input_idx = ((n * C + c) * H + ih) * W + iw;
                    int weight_idx = ((oc * C + c) * kernel_h + kh) * kernel_w + kw;
                    acc += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int out_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
    output[out_idx] = acc;
}
