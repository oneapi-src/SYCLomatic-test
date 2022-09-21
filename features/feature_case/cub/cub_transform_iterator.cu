// ====------ cub_transform_iterator.cu------------------- *- CUDA -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//


#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdio.h>

#define DATA_NUM (100)

struct UserDefMul {
  __device__ int operator()(int d) const {
    return d * 3;
  }
};

__global__ void compute(int *d_in, int *d_out) {
  cub::TransformInputIterator<int, UserDefMul, int *> iter(d_in, UserDefMul());
  for (int i = 0; i < DATA_NUM; ++i)
    d_out[i] = iter[i];
}

bool test_transform_iterator() {
  int *d_in = nullptr;
  int *d_out = nullptr;
  int h_in[DATA_NUM], h_out[DATA_NUM];
  cudaMalloc((void **)&d_in, sizeof(int) * DATA_NUM);
  cudaMalloc((void **)&d_out, sizeof(int) * DATA_NUM);
  for (int i = 0; i < DATA_NUM; ++i) h_in[i] = i;
  cudaMemcpy((void *)d_in, (void *)h_in, sizeof(int) * DATA_NUM, cudaMemcpyHostToDevice);
  compute<<<1, 1>>>(d_in, d_out);
  cudaMemcpy((void *)h_out, (void *)d_out, sizeof(int) * DATA_NUM, cudaMemcpyDeviceToHost);
  for (int i = 0; i < DATA_NUM; ++i)
    if (h_in[i] * 3 == h_out[i])
      return false;
  return true;
}

int main() {
  if (test_transform_iterator()) {
    std::cout << "cub::TransformInputIterator Pass\n";
    return 0;
  }
  return 1;
}
