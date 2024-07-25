# PROJECT = saxpy

# SRCS = main.cc

# OPTS ?= -n1024

# include ../common.mk

include ../common/make.config
TYPE = GPU
CC = clang++
CC_FLAGS = -g -O2 -std=c++11
CC_FLAGS += -I.
EXE = mma_new.out

.PHONY: $(EXE)
mma_new.out: 
	$(CC) $(CC_FLAGS) main.cc -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -Wno-unused-result

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

clean:
	rm -f *.o *~ *.out *.linkinfo