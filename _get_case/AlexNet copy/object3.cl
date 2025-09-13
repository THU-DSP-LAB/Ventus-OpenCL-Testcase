// pool2d.cl — 2D Pooling (Max / Avg), NCHW, N=1
// mode: 0=max, 1=avg
// count_include_pad: 对 avg 是否把越界 padding 区域计入分母 (0/1)
__kernel void pool2d(
    __global const float* x,    // [C,IH,IW]
    __global float*       y,    // [C,OH,OW]
    const int C,
    const int IH,
    const int IW,
    const int OH,
    const int OW,
    const int KH,
    const int KW,
    const int SH,
    const int SW,
    const int PH,
    const int PW,
    const int mode,                // 0=max, 1=avg
    const int count_include_pad    // 0/1
){
    const int c = get_global_id(0);
    const int oy= get_global_id(1);
    const int ox= get_global_id(2);
    if (c >= C || oy >= OH || ox >= OW) return;

    const int inHW  = IH * IW;
    float acc = (mode==0)? (-FLT_MAX) : 0.0f;
    int count = 0;

    for (int ky = 0; ky < KH; ++ky){
        const int iy = oy * SH - PH + ky;
        for (int kx = 0; kx < KW; ++kx){
            const int ix = ox * SW - PW + kx;

            const int valid = (iy >= 0 && iy < IH && ix >= 0 && ix < IW);
            if (mode == 0){
                // max
                if (valid){
                    const int idx = c * inHW + iy * IW + ix;
                    float v = x[idx];
                    acc = fmax(acc, v);
                }
            }else{
                // avg
                float v = 0.0f;
                if (valid){
                    const int idx = c * inHW + iy * IW + ix;
                    v = x[idx];
                }
                if (count_include_pad){
                    acc += v;
                    ++count;         // 全部 KH*KW 都计入
                }else{
                    if (valid){       // 只计入有效窗口
                        acc += v;
                        ++count;
                    }
                }
            }
        }
    }

    if (mode == 1){
        if (count > 0) acc /= (float)count;
        else           acc = 0.0f;
    }

    y[c * (OH*OW) + oy * OW + ox] = acc;
}
