__kernel void Attention(__global float *a1, __global float *a2, __global float *b, __global float *c1, __global float *c2) {
  int tid = get_global_id(0);

// 每个线程处理两个结果
// 888的Attention
 // 888矩阵乘法
    int row1 = tid/8;
    float sum1 = 0.0f;
    for (int k = 0; k < 4; ++k) {
        sum1 += a1[row1 * 8 + k] * b[k * 8 + tid % 8];
    }
    c1[tid] += sum1;

    sum1 = 0.0f;
    row1+=4;
    for (int k = 0; k < 8; ++k) {
        sum1 += a1[row1 * 8 + k] * b[k * 8 + tid % 8];
    }
    c1[tid+32] += sum1;

    float value = c1[tid];
    float value2 = c1[tid+32];
 // 数据转化为exp
    __asm__ volatile (
        "vfexp %0, %1\n"
        : "=vr"(value)               // 输出：v1 寄存器的结果赋给变量 vd
        : "vr"(c1[tid])               // 输入：vs2 寄存器包含要转换的数据
    );
    __asm__ volatile (
        "vfexp %0, %1\n"
        : "=vr"(value2)               // 输出：v1 寄存器的结果赋给变量 vd
        : "vr"(c1[tid+32])               // 输入：vs2 寄存器包含要转换的数据
    );

    c1[tid] = value;//exp 
    c1[tid+32] = value2;//exp

 // reduction: sum of exp will be saved in c1[31]; specific exp will be saved in value/value2

    for(int i=1;i<32;i*=2){
        float res=0,res2=0;

    // 使用内联汇编实现 shuffle 指令
    // .insn r opcode, func3, func7, rd, rs1, rs2
    // opcode = 1010111 (h57), func3 = 001, func7 = 0010111 (h17) (m 设置为 1，表示所有线程都有效)，vs1 = 1 表示移动1位
        __asm__ __volatile__(
            ".insn r 0x57, 0x1, 0x17, %0, %1, %2\n"//rd 输入操作数：rs1 rs2
            : "=vr"(res)           // 输出操作数：vd
            : "vr"(i),"vr"(c1[tid])//,"vr"(temp)// "I"(0x08) 
        );
        __asm__ __volatile__(
            ".insn r 0x57, 0x1, 0x17, %0, %1, %2\n"//rd 输入操作数：rs1 rs2
            : "=vr"(res2)           // 输出操作数：vd
            : "vr"(i),"vr"(c1[tid+32])//,"vr"(temp)// "I"(0x08) 
        );
        if((tid+1)%(i*2)==0) {
            c1[tid]+=res;
            c1[tid+32]+=res2;
        }
        barrier(CLK_LOCAL_MEM_FENCE); 
    }
    float sum_s1 = c1[31];
    float sum_s2 = c1[63];
 //softMax计算完成
    c1[tid] = value/sum_s1;
    c1[tid+32] = value2/sum_s2;

 // 888矩阵乘法 Attention*V
    row1 = tid/8;
    sum1 = 0.0f;
    for (int k = 0; k < 4; ++k) {
        sum1 += c1[row1 * 8 + k] * b[k * 8 + tid % 8+64];
    }
    c2[tid] += sum1;

    sum1 = 0.0f;
    row1+=4;
    for (int k = 0; k < 8; ++k) {
        sum1 += c1[row1 * 8 + k] * b[k * 8 + tid % 8+64];
    }
    c2[tid+32] += sum1;

}