// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include<cuda.h>
__global__ void mykernel_1(unsigned char *buffer, long size,
                             unsigned int *histo) {
__shared__ unsigned int temp[256];
  temp[threadIdx.x] = 0;
  __syncthreads();
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int offset = blockDim.x * gridDim.x;
  while (i < size) {
    atomicAdd(&temp[buffer[i]], 1);
    i += offset;
  }
  __syncthreads();
  atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]);
}


int main() {
return 0;
}