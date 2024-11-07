__kernel void mma(__global float *a, __global float *b, __global float *c) {
  int tid = get_global_id(0);
  float Ai = a[tid];
  float Bi = b[tid];
  float Ci = c[tid];
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  __asm__ __volatile__(".insn r 0x05, 0x0, 0x9, %0, %1, %2"
                       : "+vr"(Ci)
                       : "vr"(Bi), "vr"(Ai));
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  c[tid] = Ci;

}