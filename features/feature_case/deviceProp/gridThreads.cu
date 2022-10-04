#include <iostream>
#include <cuda_runtime.h>

#define VALUE 0x7fffffff

// ensure that maxGridSize is migrated to return an int *, and thus can be used with max/min

int f0(cudaDeviceProp *deviceProp) {
  return std::max(VALUE,deviceProp->maxGridSize[0]);
}

int f1(cudaDeviceProp *deviceProp) {
  return std::max(VALUE,deviceProp->maxGridSize[1]);
}

int f2(cudaDeviceProp *deviceProp) {
  return std::max(VALUE,deviceProp->maxGridSize[2]);
}

// ensure that maxThreadsDim is migrated to return an int *, and thus can be used with max/min

int g0(cudaDeviceProp *deviceProp) {
  return std::max(VALUE,deviceProp->maxThreadsDim[0]);
}

int g1(cudaDeviceProp *deviceProp) {
  return std::max(VALUE,deviceProp->maxThreadsDim[1]);
}

int g2(cudaDeviceProp *deviceProp) {
  return std::max(VALUE,deviceProp->maxThreadsDim[2]);
}

int main() {
  cudaDeviceProp  deviceProp;

  if (f0(&deviceProp)==VALUE  &&
      f1(&deviceProp)==VALUE  &&
      f2(&deviceProp)==VALUE  &&      

      g0(&deviceProp)==VALUE  &&
      g1(&deviceProp)==VALUE  &&
      g2(&deviceProp)==VALUE) {
    return 0;
  }
  return 1;
}
