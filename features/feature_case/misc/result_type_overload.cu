// ====------ result_type_overload.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cudnn.h>

int printResult(cudaError_t x) {
  printf("cudaError = %i\n", x);
  return 5;
}

int printResult(cudnnStatus_t x) {
  printf("cudnnStatus = %i\n", x);
  return 10;
}

int main() {
  float *d_x;
  cudnnHandle_t handle;

  cudaError_t r0;
  const cudaError_t r1 = cudaMalloc(&d_x, 5 * sizeof(*d_x));
  const cudnnStatus_t r2 = cudnnCreate(&handle);
  const int r3 = r1;
  const bool a1 = printResult(r1) == 5;
  const bool a2 = printResult(r2) == 10;
  return (a1 && a2 ? EXIT_SUCCESS : EXIT_FAILURE);
}
