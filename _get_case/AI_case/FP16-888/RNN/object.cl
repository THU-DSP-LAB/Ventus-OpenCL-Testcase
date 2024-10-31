__kernel void RNN(__global uint *a1,__global uint *a2, __global uint *b, __global uint *c1, __global uint *c2) {
  int tid = get_global_id(0);
  uint A1i = a1[tid];
  uint A2i = a2[tid];
  uint Bi = b[tid];
  uint C1i = c1[tid];
  uint C2i = c2[tid];
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  // tc.mma,shape=888,dtype=fp16 no ReLU
  __asm__ __volatile__(".insn r 0x61, 0x4, 0x0, %0, %1, %2"
                       : "=vr"(C1i)
                       : "vr"(A1i), "vr"(Bi));
  // tc.mma,shape=888,dtype=fp16 no ReLU
  __asm__ __volatile__(".insn r 0x61, 0x4, 0x0, %0, %1, %2"
                       : "=vr"(C2i)
                       : "vr"(A2i), "vr"(Bi));   
  A1i = a1[tid+32];
  A2i = a2[tid+32];
  Bi = b[tid+32];
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  // tc.mma,shape=888,dtype=fp16 ReLU
  __asm__ __volatile__(".insn r 0x61, 0x4, 0x4, %0, %1, %2"
                       : "=vr"(C1i)
                       : "vr"(A1i), "vr"(Bi));
  // tc.mma,shape=888,dtype=fp16 ReLU
  __asm__ __volatile__(".insn r 0x61, 0x4, 0x4, %0, %1, %2"
                       : "=vr"(C2i)
                       : "vr"(A2i), "vr"(Bi));                       
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);       
  c1[tid] = C1i;//+A1i+Bi;
  c2[tid] = C2i;//+A2i;

  // // 声明局部内存数组
  // __local uint sharedA[128];
  // __local uint sharedB[128];
  // __local uint sharedC[128];

  // // 将全局内存中的数据加载到局部内存
  // sharedA[tid] = Ai+1;
  // sharedB[tid] = Bi+1;
  // sharedC[tid] = Ci+1;
}