# PROJECT = AlexNet

# SRCS = AlexNet_main.cc

# OPTS ?= -n1024

include ../common/make.config
TYPE = GPU
CC = clang++
CC_FLAGS = -g -O2 -std=c++11
CC_FLAGS += -I.
EXE = AlexNet.out

.PHONY: $(EXE)
AlexNet.out: 
	$(CC) $(CC_FLAGS) AlexNet_main.cc -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -Wno-unused-result

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

clean:
	rm -f *.o *~ *.out *.linkinfo


# envs
# source /home/liuwei/ventus/env.sh
# VENTUS_BACKEND=spike    ./run # 和上一条等效
# VENTUS_BACKEND=rtlsim   ./run # 使用verilator仿真Chisel RTL
# VENTUS_BACKEND=cyclesim ./run # 使用周期精度仿真器