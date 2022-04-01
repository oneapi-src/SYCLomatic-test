// ====------ cu_kernel.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <iostream>
#include <math.h>

// CUDA kernel to add elements of two arrays
__global__
void kernel(int n, double *x, double *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
	 
	  x[i] = sin(-cos(-tan(i))); //Does not migrate
	  y[i] = __logf(__logf(i)); //Migrates successfully
	  y[i] = __logf(-__logf(i));//Does not migrate
  }
}

int main() {
 return 0; 
}
