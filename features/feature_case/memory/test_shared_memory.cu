// ====------ test_shared_memory.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda.h>

__device__ double mem_input[10];
__device__ double mem_output[10];
__global__ void kernel(double *value) {
    mem_output[threadIdx.x] = mem_input[threadIdx.x];
}
int main() {
    double *value;
    // malloc
    cudaMalloc((void**)&value, 10 * sizeof(double));
    kernel<<<1, 10>>>(value);

}