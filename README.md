# Ventus-OpenCL-Testcase
Here are some Ventus OpenCL testcases.


 'Ventus-OpenCL-Testcase/TensorCoreCase'路径下是生成硬件tensor core所需要数据的Python脚本。生成的数据与寄存器需要的数据形式保持一致。
 'Ventus-OpenCL-Testcase/_get_case'路径下包含多个OpenCL测例。在安装好LLVM-project(https://github.com/THU-DSP-LAB/llvm-project)项目后，可直接编译对应测例。
 如：
 ```
 cd _get_case/mma
 make
 ./mma.out
 ```
