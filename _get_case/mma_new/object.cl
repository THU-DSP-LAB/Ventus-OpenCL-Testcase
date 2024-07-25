__kernel void mma_new(__global float *a, __global float *b, __global float *c,
                  int n) {
  int tid = get_global_id(0);
  if (tid < n) {
    float Ai = a[tid];
    float Bi = b[tid];
    float Ci = c[tid];
    __asm__ __volatile__(".insn r 0x61, 0x4, 0x0, %0, %1, %2"
                         : "=vr"(Ci)
                         : "vr"(Bi), "vr"(Ai));
    c[tid] = Ci;
  }
}