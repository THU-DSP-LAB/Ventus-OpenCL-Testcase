__kernel void
mma(__global const float  *a,
        __global const float  *b,
        __global float  *c)
{
    //int tid = get_local_id(0);
    int tid = get_global_id(0);
    //__asm__ volatile(".insn r 0x7b, 2, 1, %0, %1, %2" : "=r"(c[tid]) : "r"(a[tid]), "r"(b[tid]));
    __asm__ volatile(
        ".insn r 0x7b, 0x4, 0x0, %0, %1, %2" 
        :
        : "+r" (c[tid]), "r" (b[tid]), "r" (a[tid]) 
    );
}
