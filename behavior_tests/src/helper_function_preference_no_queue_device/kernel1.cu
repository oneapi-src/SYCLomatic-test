#include "common.cuh"

__global__ void kernel1(int *d_Data) {}

static uint *d_Data1;

void malloc1() { cudaMalloc((void **)&d_Data1, SIZE * sizeof(int)); }

void free1() { cudaFree(d_Data1); }

void kernelWrapper1(int *d_Data) {
  kernel1<<<1, 1>>>(d_Data);
  kernel1<<<1, 1>>>(d_Data);
}
