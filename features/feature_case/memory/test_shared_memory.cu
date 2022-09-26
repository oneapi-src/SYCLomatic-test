// ====------ test_shared_memory.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda.h>
#include <stdio.h>

__device__ int mem_input[10];
__device__ int mem_output[10];
__global__ void kernel(int *value, int *out) {
    __shared__ int a;
    if (threadIdx.x == 0)
        a = 0;
    __syncthreads();
    atomicAdd(&a, 1);    
    __syncthreads();
    if (threadIdx.x == 0)
        *out = a;
    mem_output[threadIdx.x] = mem_input[threadIdx.x];
}
int main() {
    int *value;
    int *out;
    // malloc
    cudaMalloc((void**)&value, 10 * sizeof(int));
    cudaMalloc(&out, sizeof(int));
    kernel<<<1, 10>>>(value, out);
    int h;
    cudaMemcpy(&h, out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("result: %d\n", h);
    if (h == 10)
      return 0;
    return -1;
}
