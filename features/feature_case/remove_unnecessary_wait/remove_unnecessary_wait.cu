// ====------ remove_unnecessary_wait.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void kernel(float* A, float* B, float* C)
{
        C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}

int main()
{
        float A[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        float B[8] = {2, 2, 2, 2, 2, 2, 2, 2};
        float C[8];

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, 8*sizeof(float));
        cudaMalloc(&d_B, 8*sizeof(float));
        cudaMalloc(&d_C, 8*sizeof(float));

        cudaMemcpy(d_A, A, 8*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, 8*sizeof(float), cudaMemcpyHostToDevice);

        kernel<<<1, 8>>>(d_A, d_B, d_C);

        cudaMemcpy(C, d_C, 8*sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 8; i++){
          if(C[i] != 3) {
            std::cout << "test fail" << std::endl;
            exit(-1);
          }
        }
        std::cout << "test pass" << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return 0;
}