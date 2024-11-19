__kernel void shuffle_t(__global int *input,__global int *b, __global int *output) {
  int tid = get_global_id(0);
  int vs2 = input[tid];
  int vd, res=0;

  int f = 1;
  int f2 = 2;

  // 使用内联汇编实现 shuffle 指令
  // .insn r opcode, func3, func7, rd, rs1, rs2
  // opcode = 1010111 (h57), func3 = 001, func7 = 0010111 (h17) (m 设置为 1，表示所有线程都有效)，vs1 = 1 表示移动1位
  __asm__ __volatile__(
    ".insn r 0x57, 0x1, 0x17, %0, %1, %2\n"//rd 输入操作数：rs1 rs2
    : "=vr"(vd)           // 输出操作数：vd
    : "vr"(f),"vr"(vs2)
  );
  
  // 使用内联汇编实现 shuffle 指令
  // .insn r opcode, func3, func7, rd, rs1, rs2
  // opcode = 1010111 (h57), func3 = 001, func7 = 0010111 (h17) (m 设置为 1，表示所有线程都有效)，vs1 = 1 表示移动1位
  __asm__ __volatile__(
    ".insn r 0x57, 0x1, 0x17, %0, %1, %2\n"//rd 输入操作数：rs1 rs2
    : "=vr"(res)           // 输出操作数：vd
    : "vr"(f2),"vr"(vd)//,"vr"(temp)// "I"(0x08) 
  );

  output[tid] = res;
}
