// nn_forward.cl
// 全连接层前向计算内核
// 参数说明：
//   input_size   - 输入向量的长度
//   output_size  - 输出向量的长度（当前层神经元数）
//   input        - 输入向量（上一层输出或原始输入）
//   weights      - 权重矩阵，按行存储，每行对应一个神经元的权重
//   bias         - 偏置数组
//   output       - 输出向量
//   activation   - 激活标志：1 表示使用 ReLU，0 表示不激活
__kernel void fc_layer(
    const int input_size,
    const int output_size,
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    __global float* output,
    const int activation)
{
    int idx = get_global_id(0);
    if (idx < output_size) {
        float sum = 0.0f;
        // 计算内积：当前神经元对应的权重存储在 weights[idx * input_size ... ]
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[idx * input_size + i];
        }
        sum += bias[idx];
        // 若 activation 为 1，则使用 ReLU 激活
        if (activation == 1) {
            sum = fmax(sum, 0.0f);
        }
        output[idx] = sum;
    }
}
