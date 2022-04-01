// ====------ curand-device-usm.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cuda.h>
#include <curand_kernel.h>
#include <cstdio>

__global__ void my_kernel0() {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  curandState_t rng;
  curand_init(1234, tid, 0, &rng);

  for (;;) {
    float aaa = curand_uniform(&rng);
    float bbb = curand_uniform(&rng);
  }
}

__global__ void my_kernel1(unsigned long seed, curandState *rngs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, tid, 0, &rngs[tid]);
}

__global__ void my_kernel2(double *res, curandState *rngs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  res[tid] = curand_normal_double(&rngs[tid]);
}

#define MY_CHECKER(c)                                                          \
{                                                                              \
    cudaError_t error = c;                                                     \
    if (error != cudaSuccess) { printf("error!\n"); }                          \
}

int main() {
  my_kernel0<<<1, 1>>>();

  int size = 10;
  double *res;
  curandState *rngs;
  void *rngs_temp;
  cudaMalloc((void**)&rngs_temp, size * sizeof(curandState));
  rngs = (curandState*)rngs_temp;
  cudaMalloc((void**)&rngs, size * sizeof(curandState) * 10);
  cudaMalloc((void**)&rngs, size * sizeof(curandState));

  my_kernel1<<<1, 1>>>(1234, rngs);
  my_kernel2<<<1, 1>>>(res, rngs);

  int *dev_mem;
  MY_CHECKER(cudaMalloc((void **)&dev_mem, sizeof(int) * 10));
  MY_CHECKER(cudaMalloc((void **)&dev_mem, sizeof(curandState) * 10 * 10));
  dim3 grid(10, 1);
  MY_CHECKER(cudaMalloc((void **)&dev_mem, sizeof(int) * grid.x));

  return 0;
}

