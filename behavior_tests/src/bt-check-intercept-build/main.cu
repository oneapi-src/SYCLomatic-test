//===--- main.cu ------------------------------*- CUDA -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
//===------------------------------------------------------ -===//

#include <stdio.h>
#include <cuda_runtime.h>

__device__ int cal(int a) {
  if (a == 0)
    return 1;
  else
    return a * (a - 1);
}

__global__ void kfunc(int *a) {
  int id;

  id = threadIdx.x;
  a[id] = cal(id);
}

int main(void) {
  int *d;
  int h[128];

  int num;
  cudaGetDeviceCount(&num);
  cudaMalloc(&d, 128 * sizeof(int));
  kfunc<<<128, 32>>>(d);
  cudaMemcpy(h, d, 64 * sizeof(int), cudaMemcpyDeviceToHost);
  printf(MSG);
  printf("\n");
  return 0;
}
