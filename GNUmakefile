CC ?= gcc
CXX ?= g++
NVCC ?= nvcc
AR ?= ar

.DEFAULT_GOAL := build

obj:
	mkdir obj

obj/blake.o: obj
	$(CC) -c sph/blake.c -o obj/blake.o

obj/decred.o: obj
	$(NVCC) -I. -c decred.cu -o obj/decred.o

obj/cuda.a: obj/blake.o obj/decred.o
	$(AR) rvs obj/cuda.a obj/blake.o obj/decred.o

ptx:
	$(NVCC) -I. -c decred.cu --ptx

build: obj/cuda.a
	go build

install: obj/cuda.a
	go install

clean:
	rm -rf obj
	go clean
