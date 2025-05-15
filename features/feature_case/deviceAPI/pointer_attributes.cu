// ====------ pointer_attributes.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>
int main() {
  int N = 2048;
  size_t size = N * sizeof(float);

  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);

  float *d_A;
  float *d_B;
  float *d_C;
  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  void * malloc_host;
  cudaMallocHost((void **)&malloc_host, size);
  cudaPointerAttributes attributes2;
  cudaPointerGetAttributes (&attributes2, malloc_host);
  std::cout << "====== Malloc Host Attributes =======" << std::endl;
  std::cout << "malloc host " << malloc_host << std::endl;
  std::cout << attributes2.device << std::endl;
  std::cout << attributes2.hostPointer << std::endl;
  std::cout << attributes2.devicePointer << std::endl;

  cudaPointerAttributes *attributes3 = new cudaPointerAttributes();
  cudaPointerGetAttributes (attributes3, d_A);
  std::cout << "====== Device Attributes =======" << std::endl;
  std::cout << attributes3->device << std::endl;
  std::cout << attributes3->hostPointer << std::endl;
  std::cout << attributes3->devicePointer << std::endl;
  if (attributes3->type == cudaMemoryTypeHost) {
    return 0;
  } else if (attributes3->type == cudaMemoryTypeDevice) {
    return 0;
  } else if (attributes3->type == cudaMemoryTypeManaged) {
    return 0;
  } else if (attributes3->type == cudaMemoryTypeUnregistered) {
    return 0;
  }
}
