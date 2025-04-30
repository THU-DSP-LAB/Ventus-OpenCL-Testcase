__kernel void mma_new(__global uint *a, __global uint *b, __global uint *c) {
  int tid = get_global_id(0);
  uint Ai = a[tid];
  uint Bi = b[tid];
  uint Ci = c[tid];
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  // tc.mma,shape=888,dtype=fp16 无ReLU
  __asm__ __volatile__(".insn r 0x61, 0x4, 0x0, %0, %1, %2"
                       : "+vr"(Ci)
                       : "vr"(Ai), "vr"(Bi));
  // barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  // tc.mma,shape=888,dtype=fp16 ReLU
  __asm__ __volatile__(".insn r 0x61, 0x4, 0x4, %0, %1, %2"
                       : "+vr"(Ci)
                       : "vr"(Ai), "vr"(Bi));
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);       
  c[tid] = Ci;
  // tc.mma,shape=888,dtype=mixed precision no ReLU
  __asm__ __volatile__(".insn r 0x61, 0x5, 0x0, %0, %1, %2"
                       : "+vr"(Ci)
                       : "vr"(Ai), "vr"(Bi));
  __asm__ __volatile__(".insn r 0x61, 0x5, 0x0, %0, %1, %2"
                       : "+vr"(Ci)
                       : "vr"(Ai), "vr"(Bi));
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  c[tid] = Ci;
// tc.mma,shape=888,dtype=mixed precision + ReLU
  __asm__ __volatile__(".insn r 0x61, 0x5, 0x4, %0, %1, %2"
                       : "+vr"(Ci)
                       : "vr"(Ai), "vr"(Bi));
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  c[tid] = Ci;

  // 声明局部内存数组
  __local uint sharedA[128];
  __local uint sharedB[128];
  __local uint sharedC[128];

  // 将全局内存中的数据加载到局部内存
  sharedA[tid] = Ai+1;
  sharedB[tid] = Bi+1;
  sharedC[tid] = Ci+1;

}