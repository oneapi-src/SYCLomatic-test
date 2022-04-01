// ====------ hello.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdint.h>
#include <time.h>

void test() {
  cudaError_t err;
  int i = 0;
  if (err != cudaSuccess) {
    ++i;
  }

  if (err == cudaErrorAssert) {
    printf("efef");
    malloc(0x100);
  }
}

int main() {
  float *a;
  int r = cudaMalloc((void**)&a, sizeof(float));

  clock_t timer = clock();
}
