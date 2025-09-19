__kernel void conv2d_relu(
    __global const float* in,        // 输入图像
    __global const float* w,         // 卷积核
    __global float* out,             // 输出图像
    const int IC, const int IH, const int IW, // 输入的通道、高度和宽度
    const int OC, const int KH, const int KW, // 输出通道数、卷积核高宽
    const int OH, const int OW,       // 输出高度和宽度
    const int SH, const int SW,       // 步长
    const int PH, const int PW        // 填充
) {
    int oc = get_global_id(0);  // 输出通道
    int oh = get_global_id(1);  // 输出高度
    int ow = get_global_id(2);  // 输出宽度

    if (oc >= OC || oh >= OH || ow >= OW) return;

    float value = 0.0f;

    // 卷积操作
    for (int ic = 0; ic < IC; ++ic) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                int ih = oh * SH - PH + kh;  // 输入位置的高度
                int iw = ow * SW - PW + kw;  // 输入位置的宽度

                // 判断是否越界
                if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                    int in_idx = (ic * IH + ih) * IW + iw;
                    int w_idx = (oc * IC + ic) * KH * KW + kh * KW + kw;
                    value += in[in_idx] * w[w_idx];
                }
            }
        }
    }

    // ReLU 激活
    out[(oc * OH + oh) * OW + ow] = max(0.0f, value);
}
