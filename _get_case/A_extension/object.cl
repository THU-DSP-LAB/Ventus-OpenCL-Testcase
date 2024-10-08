__kernel void A_extension(__global int *data, __global int *b, __global int *c) {

  int id = get_global_id(0);
  int value=-5;
  int success=0;
  int temp;

    // 确保 data 是 4 字节对齐的
  __global int *aligned_data = (__global int *) (((uintptr_t)&data[id] + 3) & ~3);


  // 确保 data 是 4 字节对齐的
  // __global int *aligned_data = (__global int *) ((uintptr_t)data + 3) & ~3;

  // 模拟 lr.w
  // __asm__ __volatile__(
  //     "lr.w %0, (%1)"
  //     : "=r" (value)
  //     : "r" (&data[id])
  //     : "memory"
  // );
  __asm__ __volatile__(".insn r 0x2f, 0x2, 0x8, %0, %1, %2"
                      : "=vr"(value)
                      : "vr"(0x0), "vr"(&data[id])
                      : "memory");
  value = value+1;
  temp = data[id]+1;

  // 模拟 sc.w
  // __asm__ __volatile__(
  //     "sc.w %0, %2, (%1)"
  //     : "=r" (success)
  //     : "r" (&data[id]), "r" (value + 1)
  //     : "memory"
  // );
  __asm__ __volatile__(".insn r 0x2f, 0x2, 0xc, %0, %1, %2"
                  : "=vr"(success)
                  : "vr"(&temp), "vr"(value)
                  : "memory");
  // data[id]+=1;
  // // 模拟 amoswap
  __asm__ __volatile__(".insn r 0x2f, 0x2, 0x4, %0, %1, %2"
                      : "=vr"(success)
                      : "vr"(&temp), "vr"(value)
                      : "memory");

  // data[id]+=1;                    
  __asm__ __volatile__(".insn r 0x2f, 0x2, 0x0, %0, %1, %2"
                      : "=vr"(success)
                      : "vr"(&temp), "vr"(value)
                      : "memory");
  // data[id]+=1;                    
                    
  __asm__ __volatile__(".insn r 0x2f, 0x2, 0x10, %0, %1, %2"
                      : "=vr"(success)
                      : "vr"(&temp), "vr"(value)
                      : "memory");                                            
  __asm__ __volatile__(".insn r 0x2f, 0x2, 0x30, %0, %1, %2"
                      : "=vr"(success)
                      : "vr"(&temp), "vr"(value)
                      : "memory");
  __asm__ __volatile__(".insn r 0x2f, 0x2, 0x20, %0, %1, %2"
                      : "=vr"(success)
                      : "vr"(&temp), "vr"(value)
                      : "memory");
  __asm__ __volatile__(".insn r 0x2f, 0x2, 0x40, %0, %1, %2"
                      : "=vr"(success)
                      : "vr"(&temp), "vr"(value)
                      : "memory");
  __asm__ __volatile__(".insn r 0x2f, 0x2, 0x50, %0, %1, %2"
                      : "=vr"(success)
                      : "vr"(&temp), "vr"(value)
                      : "memory");
  __asm__ __volatile__(".insn r 0x2f, 0x2, 0x60, %0, %1, %2"
                      : "=vr"(success)
                      : "vr"(&temp), "vr"(value)
                      : "memory");
  __asm__ __volatile__(".insn r 0x2f, 0x2, 0x70, %0, %1, %2"
                      : "=vr"(success)
                      : "vr"(&temp), "vr"(value)
                      : "memory");
  data[id]+=1;   
  // 记录结果
  // int result[id] = (success == 0 ? 1 : 0);

}