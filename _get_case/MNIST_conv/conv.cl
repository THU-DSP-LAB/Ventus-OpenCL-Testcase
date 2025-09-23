// conv.cl
// 通用卷积 kernel
// 计算公式：
//   output[out_c, y, x] = bias[out_c] + sum_{in_c=0}^{in_channels-1} sum_{ky=0}^{kernel_h-1} sum_{kx=0}^{kernel_w-1}
//          input[in_c, y+ky, x+kx] * weight[out_c, in_c, ky, kx]
// 若 do_relu 非 0，则应用 ReLU 激活函数（即将负值置 0）

__kernel void conv(
    __global const float* input,   // 输入特征图，大小：in_channels x in_h x in_w
    __global const float* weight,  // 卷积核权重，大小：out_channels x in_channels x kernel_h x kernel_w
    __global const float* bias,    // 偏置，大小：out_channels
    __global float* output,        // 输出特征图，大小：out_channels x out_h x out_w
    const int in_channels,         // 输入通道数
    const int in_h,              // 输入高
    const int in_w,              // 输入宽
    const int kernel_h,          // 卷积核高
    const int kernel_w,          // 卷积核宽
    const int out_h,             // 输出高
    const int out_w,             // 输出宽
    const int do_relu            // 是否使用 ReLU（非 0 表示使用）
)
{
    // 使用 3D NDRange：get_global_id(0)=out_channel, get_global_id(1)=output y, get_global_id(2)=output x
    int out_c = get_global_id(0);
    int out_y = get_global_id(1);
    int out_x = get_global_id(2);

    float sum = bias[out_c];
    // 对所有输入通道及卷积核窗口求和
    for (int in_c = 0; in_c < in_channels; in_c++) {
        for (int ky = 0; ky < kernel_h; ky++) {
            for (int kx = 0; kx < kernel_w; kx++) {
                int in_y = out_y + ky;
                int in_x = out_x + kx;
                int in_index = in_c * (in_h * in_w) + in_y * in_w + in_x;
                float in_val = input[in_index];
                // weight 采用顺序：[out_c, in_c, ky, kx]
                int weight_index = out_c * (in_channels * kernel_h * kernel_w)
                                   + in_c * (kernel_h * kernel_w)
                                   + ky * kernel_w + kx;
                sum += in_val * weight[weight_index];
            }
        }
    }
    if (do_relu && sum < 0)
        sum = 0;
    int out_index = out_c * (out_h * out_w) + out_y * out_w + out_x;
    output[out_index] = sum;
}
