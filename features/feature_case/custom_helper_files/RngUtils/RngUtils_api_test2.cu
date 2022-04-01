// ====------ RngUtils_api_test2.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// CHECK: 4
// TEST_FEATURE: RngUtils_rng_generator
// TEST_FEATURE: RngUtils_rng_generator_get_engine

#include <curand_kernel.h>
__device__ void foo() {
  curandStatePhilox4_32_10_t rng;
  curand_init(1, 2, 3, &rng);
  skipahead (1, &rng);
}

int main() {
  return 0;
}
