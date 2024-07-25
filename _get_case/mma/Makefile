# include ../../common/make.config
OPENCL_DIR = ${VENTUS_INSTALL_PREFIX}/../../ocl-icd
OPENCL_INC = ${VENTUS_INSTALL_PREFIX}/include
OPENCL_LIB = ${VENTUS_INSTALL_PREFIX}/lib
POCUS_INCLUDE_DIR = /home/liuwei/pocl/poclu

#Can be changed by `make TYPE=CPU`
TYPE = GPU

#C compiler
CC = clang++

SRC = mma.c mma_exec.c

CC_FLAGS = -g -O3

EXE = mma.out

KERNEL_DIM="-DRD_WG_SIZE_0=32"

release:
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $(SRC) -o $(EXE) -I$(OPENCL_INC) -I$(POCUS_INCLUDE_DIR) -L$(OPENCL_LIB) -lOpenCL -I../util -DTIMING -Wno-unused-result

run:$(EXE)
	./mma.out

clean: 
	rm -f $(EXE)  
