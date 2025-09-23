# PROJECT = saxpy

# SRCS = main.cc

# OPTS ?= -n1024

# include ../common.mk
SHELL := /bin/bash
.ONESHELL:

include ../common/make.config
TYPE = GPU
CC = clang++
CC_FLAGS = -g -O2 -std=c++11
CC_FLAGS += -I.
EXE = matmul.out

.PHONY: $(EXE)
matmul.out: 
	$(CC) $(CC_FLAGS) main.cc -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -Wno-unused-result

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

clean:
	rm -f *.o *~ *.out *.linkinfo

# 只看完整 opcode 统计（默认）
# getSASS:
# 	python count_sass.py \
# 	/home/liuwei/Ventus-OpenCL-Testcase/_get_case/MNIST_matmul_small_timming/kernel_sm89.sass

# envs
# source /home/liuwei/ventus/env.sh
# VENTUS_BACKEND=spike    ./run # 和上一条等效
# VENTUS_BACKEND=rtlsim   ./run # 使用verilator仿真Chisel RTL
# VENTUS_BACKEND=cyclesim ./run # 使用周期精度仿真器

# # 只看“基 opcode”（把 LDG.E.128 视为 LDG）
# python count_sass.py \
#   /home/liuwei/Ventus-OpenCL-Testcase/_get_case/MNIST_matmul_small_timming/kernel_sm89.sass \
#   --mode base

# # 输出前 30 个，且按 Function 分组，并各自导出 CSV
# python count_sass.py \
#   /home/liuwei/Ventus-OpenCL-Testcase/_get_case/MNIST_matmul_small_timming/kernel_sm89.sass \
#   --by-func --top 30 --csv opcodes_global.csv --csv-per-func per_func_csv/
#  --kernel-id matmul


profiling:env_NV_profiling
	/opt/nvidia/nsight-compute/2025.2.1/ncu --metrics "flop_count_sp, inst_executed, cycle_count" ./kernel_sm89.cubin

env_NV_profiling:
	export PATH=/usr/local/cuda/bin:$$PATH
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$$LD_LIBRARY_PATH
	export PATH=/opt/nvidia/nsight-systems/2025.1.3/bin:$$PATH
	export PATH=/opt/nvidia/nsight-compute/2025.2.1/ncu:$$PATH

getbin:
	export PATH=/usr/local/cuda/bin:$$PATH
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$$LD_LIBRARY_PATH
	nvcc -arch=sm_89 -cubin kernel.ptx -o kernel_sm89.cubin

compile:
	export PATH=/usr/local/cuda/bin:$$PATH
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$$LD_LIBRARY_PATH
	nvcc -O2 -std=c++17 loader.cpp -o loader -lcuda
profile:compile
	ncu --target-processes all \
		--set full \
		--metrics sm__inst_executed.avg.per_cycle_active,sm__inst_executed.sum,sm__cycles_active.sum \
		./loader
# # nvdisasm：输出更丰富（可带行号/控制流图）
# nvdisasm --print-line-info --print-statistics kernel_sm89.cubin > kernel.sass

# # cuobjdump：也能从 cubin（或含有 fatbin 的宿主程序）中导出 SASS/PTX
# cuobjdump --dump-sass kernel_sm89.cubin > kernel.sass
# # 如需尝试提取其中的 PTX（若 fatbin 内仍带 PTX）
# cuobjdump --dump-ptx  kernel_sm89.cubin > embedded.ptx

run_cuda:
	sudo /opt/nvidia/nsight-compute/2025.2.1/ncu --target-processes all --set full --metrics sm__inst_executed.avg.per_cycle_active,sm__inst_executed.sum,sm__cycles_active.sum    ./loader


# sudo /opt/nvidia/nsight-compute/2025.2.1/ncu --kernel-name-base demangled --kernel-name ::gemm_tiled \
#     --set full \
#     --metrics sm__inst_executed.avg.per_cycle_active,sm__inst_executed.sum,sm__cycles_active.sum \
#     ./gemm