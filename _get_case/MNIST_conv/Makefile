# PROJECT = saxpy

# SRCS = main.cc

# OPTS ?= -n1024

# include ../common.mk

include ../common/make.config
TYPE = GPU
CC = clang++
CC_FLAGS = -g -O2 -std=c++11
CC_FLAGS += -I.
EXE = conv.out

.PHONY: $(EXE)
conv.out: 
	$(CC) $(CC_FLAGS) main.cc -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -Wno-unused-result

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

clean:
	rm -f *.o *~ *.out *.linkinfo

run_cuda:
	sudo /opt/nvidia/nsight-compute/2025.2.1/ncu \
    	--set full \
    	--metrics sm__inst_executed.avg.per_cycle_active,sm__inst_executed.sum,sm__cycles_active.sum \
    	./loader


# nvcc -O2 -std=c++17 loader.cpp -o loader -lcuda