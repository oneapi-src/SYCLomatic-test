// ====------ constBug.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <stdio.h>

__device__ __constant__ float const_one = 1.0;
__constant__ float A[3][3] = {
         {0.0625f, 0.125f, 0.0625f},
	 {0.1250f, 0.250f, 0.1250f},
	 {0.0625f, 0.125f, 0.0625f}};

__global__ void
constAdd(float *C)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    int k = 3*i + j;
    if (i < 3 && j < 3) {
        C[k] = A[i][j] + const_one;
    }
}

int
main(void)
{
    int size = 3*3*sizeof(float);
    
    float *h_C = (float *)malloc(size);
    float *d_C = NULL;
    cudaMalloc((void **)&d_C, size);

    constAdd<<<3,3>>>( d_C);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i  < 3; i++) {
    	for( int j = 0; j < 3; j++) {
	     sum += h_C[3*i+j];
	}
    }
    if (sum == 10.0f) {
       printf("success: %f\n",sum);
    }
    else {
       printf("fail: %f\n",sum);
    }
    return 0;
}

