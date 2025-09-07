// conv.cl — 通用 2D 卷积（stride / padding / ReLU），布局 NCHW（N=1）
__kernel void conv2d(
    __global const float* input,   // [in_c, in_h, in_w]
    __global const float* weight,  // [out_c, in_c, k_h, k_w]
    __global const float* bias,    // [out_c]
    __global float*       output,  // [out_c, out_h, out_w]
    const int in_c,
    const int in_h,
    const int in_w,
    const int out_c,
    const int k_h,
    const int k_w,
    const int out_h,
    const int out_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int do_relu
){
    const int oc = get_global_id(0);
    const int oy = get_global_id(1);
    const int ox = get_global_id(2);
    if (oc >= out_c || oy >= out_h || ox >= out_w) return;

    float sum = bias ? bias[oc] : 0.0f;

    for (int ic = 0; ic < in_c; ++ic){
        for (int ky = 0; ky < k_h; ++ky){
            const int iy = oy * stride_h - pad_h + ky;
            if (iy < 0 || iy >= in_h) continue;
            for (int kx = 0; kx < k_w; ++kx){
                const int ix = ox * stride_w - pad_w + kx;
                if (ix < 0 || ix >= in_w) continue;
                const int in_idx = ic * (in_h * in_w) + iy * in_w + ix;
                const int w_idx  = oc * (in_c * k_h * k_w)
                                 + ic * (k_h * k_w)
                                 + ky * k_w + kx;
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    if (do_relu && sum < 0.0f) sum = 0.0f;

    const int out_idx = oc * (out_h * out_w) + oy * out_w + ox;
    output[out_idx] = sum;
}
