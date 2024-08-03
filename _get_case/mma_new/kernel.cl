__kernel void mma_new(__global uint *a, __global uint *b, __global uint *c) {
  int tid = get_global_id(0);
  uint Ai = a[tid];
  uint Bi = b[tid];
  uint Ci = c[tid];
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  __asm__ __volatile__(".insn r 0x61, 0x4, 0x0, %0, %1, %2"
                       : "=vr"(Ci)
                       : "vr"(Bi), "vr"(Ai));
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  c[tid] = Ci;

}