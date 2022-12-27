// ====------ feature_profiling.cu------------- *- CUDA -* ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===-----------------------------------------------------------------===//

// includes, system
#include <stdio.h>

// includes CUDA Runtime
#include <cuda_runtime.h>

__global__ void increment_kernel(int *g_data, int inc_value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  g_data[idx] = g_data[idx] + inc_value;
}

int main() {

  int n = 256 * 256;
  int nbytes = n * sizeof(int);
  int value = 26;

  // allocate host memory
  int *a = 0;
  cudaMallocHost((void **)&a, nbytes);
  memset(a, 0, nbytes);

  // allocate device memory
  int *d_a = 0;
  cudaMalloc((void **)&d_a, nbytes);
  cudaMemset(d_a, 255, nbytes);

  // set kernel launch configuration
  dim3 threads = dim3(256, 1);
  dim3 blocks = dim3(n / threads.x, 1);

  // create cuda event handles
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaDeviceSynchronize();
  float gpu_time = 0.0f;

  // asynchronously issue work to the GPU (all to stream 0)

  cudaEventRecord(start, 0);
  cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
  increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
  cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&gpu_time, start, stop);

  // print the cpu and gpu times
  printf("time spent executing by the GPU: %.2f ms\n", gpu_time);
}
