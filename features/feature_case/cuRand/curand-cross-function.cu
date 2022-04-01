// ====------ curand-cross-function.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cuda.h>
#include <stdio.h>
#include <curand.h>

void update(float* res, curandGenerator_t rng, long long aa, long long bb) {
  curandGenerateUniform(rng, res, aa * bb);
}

int main(){
  long long aa = 1024;
  long long bb = 1024;
  unsigned long long seed = 1234ULL;
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
  curandSetPseudoRandomGeneratorSeed(rng, seed);
  float *res;
  curandGenerateUniform(rng, res, aa * bb);
  update(res, rng, aa, bb);
  curandDestroyGenerator(rng);
}

