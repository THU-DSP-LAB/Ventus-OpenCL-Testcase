# PROJECT = Softmax

# SRCS = main.cc

# OPTS ?= -n1024

include ../../common/make.config
TYPE = GPU
CC = clang++
CC_FLAGS = -g -O2 -std=c++11
CC_FLAGS += -I.
EXE = Softmax.out

.PHONY: $(EXE)
Softmax.out: 
	$(CC) $(CC_FLAGS) main.cc -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -Wno-unused-result

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

env:
	bash -c "source /home/liuwei/ventus/env.sh"

run_VENTUS: env $(EXE)
	VENTUS_BACKEND=spike ./$(EXE)

run_RTL: env $(EXE)
	VENTUS_BACKEND=rtlsim ./$(EXE)
clean:
	rm -f *.o *~ *.out *.linkinfo