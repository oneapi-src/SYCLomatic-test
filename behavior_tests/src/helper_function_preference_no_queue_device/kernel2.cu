#include "common.cuh"

__global__ void kernel2(int *d_Data) {}

static uint *d_Data2;

void malloc2() { cudaMalloc((void **)&d_Data2, SIZE * sizeof(int)); }

void free2() { cudaFree(d_Data2); }

void kernelWrapper2(int *d_Data) {
  kernel2<<<1, 1>>>(d_Data);
  kernel2<<<1, 1>>>(d_Data);
}
