// ====------ perm_byte.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//


#include <stdio.h>
#include <stdlib.h>

__global__ void byte_perm_kernel(unsigned int *d_data) {

  unsigned int lo;
  unsigned int hi;

  lo = 0x33221100;
  hi = 0x77665544;

  for (int i = 0; i < 17; i++)
    d_data[i] = __byte_perm(lo, hi, 0x1111 * i);

  d_data[17] = __byte_perm(lo, 0, 0x0123);
  d_data[18] = __byte_perm(lo, hi, 0x7531);
  d_data[19] = __byte_perm(lo, hi, 0x6420);
}

int main(void) {

  unsigned int *d_data;
  const int N = 20;
  unsigned int mem_size = N * sizeof(unsigned int);
  cudaMalloc((void **)&d_data, mem_size);
  unsigned int *h_data = (unsigned int *)malloc(mem_size);
  unsigned int ref_data[N] = {0x0,        0x11111111, 0x22222222, 0x33333333,
                              0x44444444, 0x55555555, 0x66666666, 0x77777777,
                              0x0,        0x11111111, 0x22222222, 0x33333333,
                              0x44444444, 0x55555555, 0x66666666, 0x77777777,
                              0x11111100, 0x112233,   0x77553311, 0x66442200};

  byte_perm_kernel<<<1, 1>>>(d_data);
  cudaDeviceSynchronize();

  cudaMemcpy(h_data, d_data, mem_size, cudaMemcpyDeviceToHost);

  bool pass = true;
  for (int i = 0; i < N; i++) {
    if (h_data[i] != ref_data[i])
      pass = false;
  }

  free(h_data);
  cudaFree(d_data);

  if (pass) {
    printf("Test passed!\n");
    return 0;
  } else {
    printf("Test failed!\n");
    return -1;
  }
}
