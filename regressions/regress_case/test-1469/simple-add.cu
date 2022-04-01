// ====------ simple-add.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cuda.h>
#include <stdio.h>

const int vector_size = 256;

__global__ void SimpleAddKernel(float *A, int offset) 
{
  A[threadIdx.x] = threadIdx.x + offset;
}

int main() 
{
  float *d_A;
  int offset = 10000;

  cudaMalloc( &d_A, vector_size * sizeof( float ) );
  SimpleAddKernel<<<1, vector_size>>>(d_A, offset);

  float result[vector_size] = { };
  cudaMemcpy(result, d_A, vector_size*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree( d_A );
   
  for (int i = 0; i < vector_size; ++i) {
    if (i % 8 == 0) printf( "\n" );
    printf( "%.1f ", result[i] );
  }

  return 0;
}

