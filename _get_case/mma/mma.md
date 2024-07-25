
__kernel void
vecadd (__global const float *a,
	__global const float *b,
	__global float *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] + b[gid];
  in

    asm volatile(“.insn r 0x7b, 2, 1, x0, %1, x0” : “=r”(zero) : “r”(addr));

}


__global__ void f64mma1688NaiveKernel(const double *__restrict__ A, const double *__restrict__ B,
                                      double *__restrict__ C)
{

    const uint32_t laneId = threadIdx.x % WARP_SIZE;

    double RA[4];
    double RB[2];
    double RC[4] = {0, 0, 0, 0};

    RA[0] = A[laneId / 4 * WMMA_K + laneId % 4];
    RA[1] = A[8 * WMMA_K + laneId / 4 * WMMA_K + laneId % 4];
    RA[2] = A[laneId / 4 * WMMA_K + laneId % 4 + 4];
    RA[3] = A[8 * WMMA_K + laneId / 4 * WMMA_K + laneId % 4 + 4];

    RB[0] = B[laneId / 4 * WMMA_K + laneId % 4];
    RB[1] = B[8 * WMMA_K + laneId / 4 * WMMA_K + laneId % 4];

 
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=d"(RC[0]), "=d"(RC[1]), "=d"(RC[2]), "=d"(RC[3])
                 : "d"(RA[0]), "d"(RA[1]), "d"(RA[2]), "d"(RA[3]), "d"(RB[0]), "d"(RB[1]), "d"(RC[0]), "d"(RC[1]), "d"(RC[2]), "d"(RC[3]));

    C[laneId / 4 * WMMA_N + laneId % 4 * 2] = RC[0];
    C[laneId / 4 * WMMA_N + laneId % 4 * 2 + 1] = RC[1];
    C[8 * WMMA_N + laneId / 4 * WMMA_N + laneId % 4 * 2] = RC[2];
    C[8 * WMMA_N + laneId / 4 * WMMA_N + laneId % 4 * 2 + 1] = RC[3];
}