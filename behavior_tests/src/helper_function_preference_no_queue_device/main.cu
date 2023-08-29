#include "common.cuh"
#include <stdio.h>

int main() {
  int *h_Data;
  int *d_Data;
  cudaDeviceProp deviceProp;
  deviceProp.major = 0;
  cudaGetDeviceProperties(&deviceProp, 0);
  h_Data = (int *)malloc(SIZE * sizeof(int));
  cudaMalloc((void **)&d_Data, SIZE * sizeof(int));
  malloc1();
  kernelWrapper1(d_Data);
  cudaDeviceSynchronize();
  cudaMemcpy(h_Data, d_Data, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
  free1();
  malloc2();
  kernelWrapper2(d_Data);
  cudaDeviceSynchronize();
  cudaMemcpy(h_Data, d_Data, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
  free2();
  cudaFree(d_Data);
  free(h_Data);
  printf("test pass!\n");
  return 0;
}
