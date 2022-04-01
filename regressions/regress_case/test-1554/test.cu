// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <assert.h>
 #include <stdlib.h>
 #include <time.h>
#include "cuda_runtime.h"
 #include "curand.h"
 #include "cublas_v2.h"
 #include <assert.h>
void cuda_random(float *x_gpu, size_t n)
 {
 static curandGenerator_t gen[16];
 static int init[16] = {0};
 int i = 0;
 if(!init[i])
{ curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT); curandSetPseudoRandomGeneratorSeed(gen[i], time(0)); init[i] = 1; }
curandGenerateUniform(gen[i], x_gpu, n);
 }
 
int main() {
return 0;
}
