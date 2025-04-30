__kernel void Attention(__global float *a1, __global float *a2, __global float *b, __global float *c1, __global float *c2) {
  int tid = get_global_id(0);
  uint A1i = a1[tid];
  uint A2i = a2[tid];
  uint Bi = b[tid];
  uint C1i = c1[tid];
  uint C2i = c2[tid];


// 每个线程处理两个结果
// 888的Attention
 // 888矩阵乘法
   // tc.mma,shape=888,dtype=fp16 no ReLU
    // __asm__ __volatile__(".insn r 0x61, 0x4, 0x0, %0, %1, %2"
    //                    : "+vr"(C1i)
    //                    : "vr"(A1i), "vr"(Bi));
    // tc.mma,shape=848,dtype=fp16 no ReLU
  __asm__ __volatile__(".insn r 0x61, 0x4, 0x2, %0, %1, %2"
                       : "+vr"(C1i)
                       : "vr"(A1i), "vr"(Bi));
// tc.mma,shape=848,dtype=fp16 no ReLU
  __asm__ __volatile__(".insn r 0x61, 0x4, 0x2, %0, %1, %2"
                       : "+vr"(c1[tid+1])
                       : "vr"(a1[tid+1]), "vr"(Bi));
    barrier(CLK_LOCAL_MEM_FENCE); 
 // 数据精度转换 Half->float
    uint high16;      // 保存 a 的高 16 位
    uint low16;
    high16 = ((C1i >> 16) & 0xFFFF);
    low16 = ((C1i) & 0xFFFF);
    __asm__ __volatile__ (
      "vfcvt.f.xu.v %0, %1\n"   // 将浮点向量 v2 转换为整数向量 v1
      : "=vr"(c1[tid+32])               // 输出：v1 寄存器的结果赋给变量 vd
      : "vr"(high16)               // 输入：vs2 寄存器包含要转换的数据
    );
    
    __asm__ __volatile__ (
      "vfcvt.f.xu.v %0, %1\n"   // 将浮点向量 v2 转换为整数向量 v1
      : "=vr"(c1[tid])               // 输出：v1 寄存器的结果赋给变量 vd
      : "vr"(low16)               // 输入：vs2 寄存器包含要转换的数据
    );

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

 // 精度转换
    uint high016,low016;

    __asm__ __volatile__ (
      "vfcvt.x.f.v %0, %1\n"   // 将浮点向量 v2 转换为整数向量 v1
      : "=vr"(high016)               // 输出：v1 寄存器的结果赋给变量 vd
      : "vr"(c1[tid+32])               // 输入：vs2 寄存器包含要转换的数据
      // : "v1", "v2"             // 通知编译器 v1 和 v2 寄存器将被修改
    );

    __asm__ __volatile__ (
      "vfcvt.x.f.v %0, %1\n"   // 将浮点向量 v2 转换为整数向量 v1
      : "=vr"(low016)               // 输出：v1 寄存器的结果赋给变量 vd
      : "vr"(c1[tid])               // 输入：vs2 寄存器包含要转换的数据
      // : "v1", "v2"             // 通知编译器 v1 和 v2 寄存器将被修改
    );

    uint temp_add = (high016<<16) + low016;
    c1[tid] = temp_add;

 // 888矩阵乘法 Attention*V
    // tc.mma,shape=888,dtype=fp16 no ReLU
    // __asm__ __volatile__(".insn r 0x61, 0x4, 0x0, %0, %1, %2"
    //                    : "+vr"(C2i)
    //                    : "vr"(C1i), "vr"(Bi));
    // tc.mma,shape=848,dtype=fp16 no ReLU
  __asm__ __volatile__(".insn r 0x61, 0x4, 0x2, %0, %1, %2"
                       : "+vr"(C1i)
                       : "vr"(c1[tid]), "vr"(Bi));
// tc.mma,shape=848,dtype=fp16 no ReLU
  __asm__ __volatile__(".insn r 0x61, 0x4, 0x2, %0, %1, %2"
                       : "+vr"(C2i)
                       : "vr"(c1[tid+1]), "vr"(Bi));


// 888的Attention 2
 // 888矩阵乘法
   // tc.mma,shape=888,dtype=fp16 no ReLU
    // __asm__ __volatile__(".insn r 0x61, 0x4, 0x0, %0, %1, %2"
    //                    : "+vr"(C1i)
    //                    : "vr"(A1i), "vr"(Bi));
    // tc.mma,shape=848,dtype=fp16 no ReLU
  __asm__ __volatile__(".insn r 0x61, 0x4, 0x2, %0, %1, %2"
                       : "+vr"(c1[tid+128])
                       : "vr"(a1[tid+128]), "vr"(b[tid+128]));
// tc.mma,shape=848,dtype=fp16 no ReLU
  __asm__ __volatile__(".insn r 0x61, 0x4, 0x2, %0, %1, %2"
                       : "+vr"(c1[tid+1+128])
                       : "vr"(a1[tid+1+128]), "vr"(b[tid+128]));
    barrier(CLK_LOCAL_MEM_FENCE); 
 // 数据精度转换 Half->float
 C1i = c1[tid+128];


    high16 = ((C1i >> 16) & 0xFFFF);
    low16 = ((C1i) & 0xFFFF);
    __asm__ __volatile__ (
      "vfcvt.f.xu.v %0, %1\n"   // 将浮点向量 v2 转换为整数向量 v1
      : "=vr"(c1[tid+32+128])               // 输出：v1 寄存器的结果赋给变量 vd
      : "vr"(high16)               // 输入：vs2 寄存器包含要转换的数据
    );
    
    __asm__ __volatile__ (
      "vfcvt.f.xu.v %0, %1\n"   // 将浮点向量 v2 转换为整数向量 v1
      : "=vr"(c1[tid+128])               // 输出：v1 寄存器的结果赋给变量 vd
      : "vr"(low16)               // 输入：vs2 寄存器包含要转换的数据
    );

    value = c1[tid+128];
    value2 = c1[tid+32+128];
 // 数据转化为exp
    __asm__ volatile (
        "vfexp %0, %1\n"
        : "=vr"(value)               // 输出：v1 寄存器的结果赋给变量 vd
        : "vr"(c1[tid+128])               // 输入：vs2 寄存器包含要转换的数据
    );
    __asm__ volatile (
        "vfexp %0, %1\n"
        : "=vr"(value2)               // 输出：v1 寄存器的结果赋给变量 vd
        : "vr"(c1[tid+32+128])               // 输入：vs2 寄存器包含要转换的数据
    );

    c1[tid+128] = value;//exp 
    c1[tid+32+128] = value2;//exp

 // reduction: sum of exp will be saved in c1[31]; specific exp will be saved in value/value2

    for(int i=1;i<32;i*=2){
        float res=0,res2=0;

    // 使用内联汇编实现 shuffle 指令
    // .insn r opcode, func3, func7, rd, rs1, rs2
    // opcode = 1010111 (h57), func3 = 001, func7 = 0010111 (h17) (m 设置为 1，表示所有线程都有效)，vs1 = 1 表示移动1位
        __asm__ __volatile__(
            ".insn r 0x57, 0x1, 0x17, %0, %1, %2\n"//rd 输入操作数：rs1 rs2
            : "=vr"(res)           // 输出操作数：vd
            : "vr"(i),"vr"(c1[tid+128])//,"vr"(temp)// "I"(0x08) 
        );
        __asm__ __volatile__(
            ".insn r 0x57, 0x1, 0x17, %0, %1, %2\n"//rd 输入操作数：rs1 rs2
            : "=vr"(res2)           // 输出操作数：vd
            : "vr"(i),"vr"(c1[tid+32+128])//,"vr"(temp)// "I"(0x08) 
        );
        if((tid+1)%(i*2)==0) {
            c1[tid+128]+=res;
            c1[tid+32+128]+=res2;
        }
        barrier(CLK_LOCAL_MEM_FENCE); 
    }
    sum_s1 = c1[31+128];
    sum_s2 = c1[63+128];
 //softMax计算完成
    c1[tid+128] = value/sum_s1;
    c1[tid+32+128] = value2/sum_s2;

 // 精度转换
    __asm__ __volatile__ (
      "vfcvt.x.f.v %0, %1\n"   // 将浮点向量 v2 转换为整数向量 v1
      : "=vr"(high016)               // 输出：v1 寄存器的结果赋给变量 vd
      : "vr"(c1[tid+32+128])               // 输入：vs2 寄存器包含要转换的数据
      // : "v1", "v2"             // 通知编译器 v1 和 v2 寄存器将被修改
    );

    __asm__ __volatile__ (
      "vfcvt.x.f.v %0, %1\n"   // 将浮点向量 v2 转换为整数向量 v1
      : "=vr"(low016)               // 输出：v1 寄存器的结果赋给变量 vd
      : "vr"(c1[tid+128])               // 输入：vs2 寄存器包含要转换的数据
      // : "v1", "v2"             // 通知编译器 v1 和 v2 寄存器将被修改
    );

    temp_add = (high016<<16) + low016;
    c1[tid+128] = temp_add;

 // 888矩阵乘法 Attention*V
    // tc.mma,shape=888,dtype=fp16 no ReLU
    // __asm__ __volatile__(".insn r 0x61, 0x4, 0x0, %0, %1, %2"
    //                    : "+vr"(C2i)
    //                    : "vr"(C1i), "vr"(Bi));
    // tc.mma,shape=848,dtype=fp16 no ReLU
  __asm__ __volatile__(".insn r 0x61, 0x4, 0x2, %0, %1, %2"
                       : "+vr"(c1[tid+128])
                       : "vr"(c1[tid+128]), "vr"(b[tid+128]));
// tc.mma,shape=848,dtype=fp16 no ReLU
  __asm__ __volatile__(".insn r 0x61, 0x4, 0x2, %0, %1, %2"
                       : "+vr"(c2[tid+128])
                       : "vr"(c1[tid+1+128]), "vr"(b[tid+128]));

}