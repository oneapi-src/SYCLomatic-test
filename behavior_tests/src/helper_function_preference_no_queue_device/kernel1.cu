// ====-------------- kernel1.cu ----------- *- CUDA -* -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include "common.cuh"

__global__ void kernel1(int *d_Data) {}

static uint *d_Data1;

void malloc1() { cudaMalloc((void **)&d_Data1, SIZE * sizeof(int)); }

void free1() { cudaFree(d_Data1); }

void kernelWrapper1(int *d_Data) {
  kernel1<<<1, 1>>>(d_Data);
  kernel1<<<1, 1>>>(d_Data);
}
