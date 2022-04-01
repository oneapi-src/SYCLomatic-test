// ====------ atomicWithinDifferentNS.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <stdio.h>
#include <device_atomic_functions.h>

namespace yakl {
  __device__ int atomicAdd(char * address, int value)
  {
    
	  return ::atomicAdd((unsigned int*)address, value);
  }
};

__global__ void ker(char *count)
{
    int n=1;
    int x = yakl::atomicAdd (&count[0],n);
    printf("In kernel count is %d\n", x);
}

int main()
{
    char hitCount[1];
    char *hitCount_d;

    hitCount[0]=1;
    cudaMalloc((void **)&hitCount_d,1*sizeof(char));

    cudaMemcpy(&hitCount_d[0],&hitCount[0],1*sizeof(char),cudaMemcpyHostToDevice);

    ker<<<1,4>>>(hitCount_d);

    cudaMemcpy(&hitCount[0],&hitCount_d[0],1*sizeof(char),cudaMemcpyDeviceToHost);

    printf("On HOST, count is %c\n",hitCount[0]);
    
    return 0;
}
