//__constant__ array declared in header must be migrated to a static variable
#include "constant_array.cuh"

void init_h1(int *hvals) {
  cudaMemcpyToSymbol(dvals, hvals, 2 * sizeof(int), 0, cudaMemcpyHostToDevice);
}

void get_h1(int *target) {
  cudaMemcpyFromSymbol(target, dvals, 2 * sizeof(int), 0, cudaMemcpyDeviceToHost);
}
