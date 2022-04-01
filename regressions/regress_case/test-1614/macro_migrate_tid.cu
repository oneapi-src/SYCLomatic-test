// ====------ macro_migrate_tid.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <iostream>
//#include <math.h>
#define MUL(a, b) __umul24(a, b)
#define LOG(a) __logf(a)

__global__
void kernel(int n, double *x, double *y)
{
   unsigned int tid = MUL(blockDim.x, blockIdx.x) + threadIdx.x;
   unsigned int  threadN = MUL(blockDim.x, gridDim.x);	
  for (unsigned int i = tid; i < n; i += threadN) {
	 
	  x[i] = sin(-cos(-tan(i))); //Did not migrate until beta06
	  y[i] = MUL(x[i], y[i]);
	  x[i] = LOG(x[i]);
  }
}


int main() {
return 0;
}
