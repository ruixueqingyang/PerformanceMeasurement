CUDA_DIR = /opt/cuda
CUDA_INC = $(CUDA_DIR)/include
CUDA_GDK = $(CUDA_DIR)/gdk
CUDA_LIB = $(CUDA_DIR)/lib64

CC = g++
CFLAGS = -Wall -g -O0
SPECIALFLAGS = -lnvidia-ml -lpthread

SRC = main.cpp
TARGET = PerfMeasure

all:  $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -I$(CUDA_INC) -I$(CUDA_GDK) -L$(CUDA_LIB) -lcudart -lcuda $(SPECIALFLAGS) -o $@.bin $(SRC)

gdb:
	gdb ./$(TARGET).bin

clean:
	rm -f *.o *.bin