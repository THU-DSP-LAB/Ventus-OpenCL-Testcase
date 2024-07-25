__kernel void dma_3(__global float* input, __global float* output){//, __local float* local_mem) {
    __local float sharedmem1[128];
    __local float sharedmem2[128];
    // __local float* shared;
    // shared = sharedmem1;
    int size = 64 * 4;
    int lid = get_local_id(0);
    // float in_reg = input[0];

    // int cube = 10;
    // int addr = 11;
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if(lid == 0){
        __asm__ __volatile__(
            ".insn r 0x41, 0x1, 0x0, %0, %1, %2"
            :
            :"vr"(sharedmem1),"r"(input),"r"(size)
        );
        // __asm__ volatile(".insn r 0x7b, 6, 6, %0, %1, x0" , "=r"(cube) , "r"(addr));
    }
    if(lid == 64){
        __asm__ __volatile__(
            ".insn r 0x41, 0x1, 0x0, %0, %1, %2"
            :
            :"vr"(sharedmem2),"r"(input + size / 4),"r"(size)
        );
    }
    // for(int i = 0 ; i < index; i ++){
    //     sharedmem2[index * lid + i] = input[index * lid + i];
    // }
    // barrier(CLK_LOCAL_MEM_FENCE);
    // 将全局内存中的数据拷贝到局部内存中
    __asm__ __volatile__(
            ".insn r 0x41, 0x4, 0x0, %0, %1, %2"
            :
            :"vr"(sharedmem1),"r"(input),"r"(size)
        );
    // float res1 = 0.5;
    // if(lid % 64 < 64 && lid / 64 == 0){
    //     res1 = sharedmem1[lid]
    // }

    // __asm__ __volatile__(
    //         ".insn r 0x41, 0x4, 0x0, %0, %1, %2"
    //         :
    //         :"vr"(sharedmem1),"r"(input + size),"r"(size)
    //     );

    
    // 等待所有工作项完成局部内存的拷贝
    // barrier(CLK_LOCAL_MEM_FENCE);

    // 对局部内存中的数据进行简单操作
    int i = 0;
    float data = 0.61;
    if(lid < 64)
    {
        while (i < 50)
        {
            sharedmem1[lid] = (data - 9) * (i % 19) * sharedmem2[lid] + i;
            i = i + 1;
        }
    }
     i = 0;
    if(lid >= 64){
        while (i < 50)
        {
            sharedmem2[lid - 64] =(data - 7) * (i % 19) * i * sharedmem1[lid - 64] + i/ 6.3;
            i = i + 1;
        }
    }
    output[lid] = sharedmem1[lid]* sharedmem2[lid] * 2.0f;
}
