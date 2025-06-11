// 通用卷积 kernel（支持 stride）
// 计算公式：
// output[out_c, y, x] = bias[out_c] + sum_{in_c, ky, kx} input[in_c, y*stride+ky, x*stride+kx] * weight[out_c, in_c, ky, kx]

__kernel void conv(
    __global const float* input,   // 输入特征图，大小：in_channels x in_h x in_w
    __global const float* weight,  // 卷积核权重，大小：out_channels x in_channels x kernel_h x kernel_w
    __global const float* bias,    // 偏置，大小：out_channels
    __global float* output,        // 输出特征图，大小：out_channels x out_h x out_w
    const int in_channels,         // 输入通道数
    const int in_h,                // 输入高
    const int in_w,                // 输入宽
    const int kernel_h,            // 卷积核高
    const int kernel_w,            // 卷积核宽
    const int out_h,               // 输出高
    const int out_w,               // 输出宽
    const int do_relu,             // 是否使用 ReLU
    const int stride_h,            // 高方向 stride
    const int stride_w             // 宽方向 stride
)
{
    // 使用 3D NDRange：get_global_id(0)=out_channel, get_global_id(1)=output y, get_global_id(2)=output x
    int out_c = get_global_id(0);
    int out_y = get_global_id(1);
    int out_x = get_global_id(2);

    float sum = bias[out_c];

    for (int in_c = 0; in_c < in_channels; in_c++) {
        for (int ky = 0; ky < kernel_h; ky++) {
            for (int kx = 0; kx < kernel_w; kx++) {
                int in_y = out_y * stride_h + ky;
                int in_x = out_x * stride_w + kx;

                if (in_y >= in_h || in_x >= in_w)
                    continue; // 边界检查（防止越界）

                int in_index = in_c * (in_h * in_w) + in_y * in_w + in_x;
                int weight_index = out_c * (in_channels * kernel_h * kernel_w)
                                 + in_c * (kernel_h * kernel_w)
                                 + ky * kernel_w + kx;
                sum += input[in_index] * weight[weight_index];
            }
        }
    }

    if (do_relu && sum < 0)
        sum = 0;

    int out_index = out_c * (out_h * out_w) + out_y * out_w + out_x;
    output[out_index] = sum;
}
