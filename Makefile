NVCC=nvcc
LDLIBS=
TARGET=gpu-des
DEBUG=
CCFLAGS=-Wextra -Wall -Wunused-parameter -O2
CUDA_COMPILER_OPTIONS=$(addprefix --compiler-options ,${CCFLAGS}) --std=c++11 -dc -O2
ALL_CCFLAGS=
ALL_LDFLAGS=
GENCODE_FLAGS=--gpu-architecture=compute_35
INCLUDES=-I/opt/cuda/samples/common/inc -I/usr/local/cuda/samples/common/inc

.PHONY: all clean debug

all: ${TARGET}

debug: DEBUG+=-G
debug: ${TARGET}

OBJECTS = main.o des_cpu.o

des_cpu.o: des_cpu.cpp des_cpu.h 
	${NVCC} ${INCLUDES} ${DEBUG} ${GENCODE_FLAGS} ${CUDA_COMPILER_OPTIONS} -c $< -o $@

main.o: main.cpp des_cpu.h
	${NVCC} ${INCLUDES} ${GENCODE_FLAGS} ${CUDA_COMPILER_OPTIONS} -c $< -o $@

${TARGET}: ${OBJECTS}
	${NVCC} ${GENCODE_FLAGS} $^ -o $@

clean:
	rm -Iv ${TARGET} *.o