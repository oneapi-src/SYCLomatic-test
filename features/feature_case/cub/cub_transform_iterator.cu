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
#define EPS (1e-6)

struct UserDefMul {
  __device__ double operator()(double d) const {
    return d * 3.0;
  }
};

__global__ void compute(double *d_in, double *d_out) {
  cub::TransformInputIterator<double, UserDefMul, double *> iter(d_in, UserDefMul());
  for (int i = 0; i < DATA_NUM; ++i)
    d_out[i] = iter[i];
}

void print_array(double *d) {
  for (int i = 0; i < DATA_NUM; ++i)
    printf("%.2lf ", d[i]);
  printf("\n");
}

bool equal(double a, double b) {
  return a - b > -EPS && a - b < EPS;
}

bool test_transform_iterator() {
  double *d_in = nullptr;
  double *d_out = nullptr;
  double h_in[DATA_NUM], h_out[DATA_NUM];
  cudaMalloc((void **)&d_in, sizeof(double) * DATA_NUM);
  cudaMalloc((void **)&d_out, sizeof(double) * DATA_NUM);
  for (int i = 0; i < DATA_NUM; ++i) h_in[i] = i;
  cudaMemcpy((void *)d_in, (void *)h_in, sizeof(double) * DATA_NUM, cudaMemcpyHostToDevice);
  compute<<<1, 1>>>(d_in, d_out);
  cudaMemcpy((void *)h_out, (void *)d_out, sizeof(double) * DATA_NUM, cudaMemcpyDeviceToHost);
  for (int i = 0; i < DATA_NUM; ++i)
    if (!equal(h_in[i] * 3.0, h_out[i]))
      return false;
  return true;
}

int main() {
  if (test_transform_iterator())
    std::cout << "cub::TransformInputIterator Pass\n";
  return 0;
}
