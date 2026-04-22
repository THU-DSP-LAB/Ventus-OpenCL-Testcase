VENTUS_ROOT ?= $(abspath ../../..)
CC ?= cc
CFLAGS += -DCL_TARGET_OPENCL_VERSION=200 -I$(VENTUS_ROOT)/install/include
LDFLAGS += -L$(VENTUS_ROOT)/install/lib
LDLIBS += -lOpenCL

TARGET := lds_stomp_suite.out
SRC := main.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) $(LDLIBS) -o $@

clean:
	rm -f $(TARGET)
