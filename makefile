OPT_CUDA_DIR = /opt/cuda
ifeq (${OPT_CUDA_DIR}, ${wildcard ${OPT_CUDA_DIR}})
#目录存在
	CUDA_DIR = ${OPT_CUDA_DIR}
else
#目录不存在
endif

LOCAL_CUDA_DIR = /usr/local/cuda
ifeq (${LOCAL_CUDA_DIR}, ${wildcard ${LOCAL_CUDA_DIR}})
#目录存在
	CUDA_DIR = ${LOCAL_CUDA_DIR}
endif

CUDA_INC = $(CUDA_DIR)/include
CUDA_GDK = $(CUDA_DIR)/gdk
CUDA_LIB = $(CUDA_DIR)/lib64

# POWER_MANAGER_DIR := /home/wfr/work/Energy/PowerManager
# POWER_MANAGER_SRC := ${POWER_MANAGER_DIR}/PowerManager.cpp

CC = g++
CFLAGS = -Wall -g -O0
SPECIALFLAGS = -lnvidia-ml -lpthread

SRC = main.cpp PowerManager.cpp
TARGET = PerfMeasure

all:  $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -I$(CUDA_INC) -I$(CUDA_GDK) -L$(CUDA_LIB) -lcudart -lcuda $(SPECIALFLAGS) -o $@.bin $(SRC)

mytest:
	$(warning, "test00")
	$(warning, "${CUDA_DIR}")
	@echo "${CUDA_DIR}"

gdb:
	gdb ./$(TARGET).bin

clean:
	rm -f *.o *.bin