__kernel void cvt(__global int *input,__global int *b, __global int *output) {
  int tid = get_global_id(0);
  int vs2 = input[tid];
  int vd;
  int half_res;
  uint temp = 8;
  uint temp2 = 9;
  // 0	1	0	0	1	0	m	vs2					0	1	0	0	0	0	0	1	vd					1	0	1	0	1	1	1	vfcvt.f.h.v vd, vs2, vm	Half -> Float
  // 0	1	0	0	1	0	m	vs2					0	1	0	0	1	0	0	1	vd					1	0	1	0	1	1	1	vfcvt.h.f.v vd, vs2, vm	Float -> Half

  __asm__ __volatile__ (
      "vfcvt.f.xu.v %0, %1\n"   // 将浮点向量 v2 转换为整数向量 v1
      : "=vr"(vd)               // 输出：v1 寄存器的结果赋给变量 vd
      : "vr"(vs2)               // 输入：vs2 寄存器包含要转换的数据
      // : "v1", "v2"             // 通知编译器 v1 和 v2 寄存器将被修改
  );

  // 使用内联汇编实现 vfcvt.f.h.v 指令， Half -> Float
  // .insn r opcode, func3, func7, rd, rs1, rs2
  // opcode = 1010111（57）, func3 = 001, func7 = 0100101 (m 设置为 1，表示所有线程都有效)，vs1 = 01000
  __asm__ __volatile__(
    ".insn r 0x57, 0x1, 0x25, %0, v8, %1\n"
    : "=vr"(vd)           // 输出操作数：vd
    : "vr"(vs2)//,"vr"(vs2)// "I"(0x08) // 输入操作数：vs1, vs2
  );

  // __asm__ __volatile__(
  //   ".insn r 0x57, 0x1, 0x25, %0, %1, %2\n"
  //   : "=vr"(vd)           // 输出操作数：vd
  //   : "vr"(0x9),"vr"(vs2)// "I"(0x08) // 输入操作数：vs1, vs2
  // );
  
  output[tid] = vd;
  __asm__ __volatile__ (
      "vfcvt.x.f.v %0, %1\n"   // 将浮点向量 v2 转换为整数向量 v1
      : "=vr"(half_res)               // 输出：v1 寄存器的结果赋给变量 vd
      : "vr"(vd)               // 输入：vs2 寄存器包含要转换的数据
      // : "v1", "v2"             // 通知编译器 v1 和 v2 寄存器将被修改
  );
  // 使用内联汇编实现 vfcvt.h.f.v 指令， Float -> Half
  // .insn r opcode, func3, func7, rd, rs1, rs2
  // opcode = 1010111（57）, func3 = 001, func7 = 0100101 (m 设置为 1，表示所有线程都有效)，vs1 = 01001
  __asm__ __volatile__(
    ".insn r 0x57, 0x1, 0x25, %0, v9, %1\n"
    : "=vr"(half_res)           // 输出操作数：vd
    : "vr"(vd)//,"vr"(vd)//, "I"(0x09) // 输入操作数：vs2, vs1
  );

  output[tid] = half_res;
}
