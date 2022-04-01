// ====------ curand-device.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cuda.h>
#include <curand_kernel.h>

__global__ void my_kernel0() {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  curandStatePhilox4_32_10_t rng;
  curand_init(1234, tid, 10, &rng);

  for (;;) {
    float2 aaa = curand_normal2(&rng);
    float2 bbb = curand_normal2(&rng);
  }
}

__global__ void my_kernel1(unsigned long seed, curandStateMRG32k3a_t *rngs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, tid, 10, &rngs[tid]);
}

__global__ void my_kernel2(double2 *res, curandStateMRG32k3a_t *rngs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  res[tid] = curand_normal2_double(&rngs[tid]);
}

int main() {
  my_kernel0<<<1, 1>>>();

  int size = 10;
  double2 *res;
  curandStateMRG32k3a_t *rngs;
  cudaMalloc((void**)&rngs, size * sizeof(curandStateMRG32k3a_t));

  my_kernel1<<<1, 1>>>(1234, rngs);
  my_kernel2<<<1, 1>>>(res, rngs);

  return 0;
}

int foo() {
  int size = 10;
  curandStateMRG32k3a_t *rngs;
  cudaMalloc((void**)&rngs, size * sizeof(curandStateMRG32k3a_t));
  my_kernel1<<<64, 128>>>(1234, rngs);
  return 0;
}

