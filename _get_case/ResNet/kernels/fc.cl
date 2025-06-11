// fc.cl

__kernel void fc_forward(
    __global const float* input,   // [N, in_features]
    __global const float* weight,  // [out_features, in_features]
    __global const float* bias,    // [out_features]
    __global float* output,        // [N, out_features]
    const int N,
    const int in_features,
    const int out_features
) {
    int n = get_global_id(0); // batch index
    int o = get_global_id(1); // output feature index

    if (n < N && o < out_features) {
        float acc = bias[o];
        for (int i = 0; i < in_features; ++i) {
            acc += input[n * in_features + i] * weight[o * in_features + i];
        }
        output[n * out_features + o] = acc;
    }
}
