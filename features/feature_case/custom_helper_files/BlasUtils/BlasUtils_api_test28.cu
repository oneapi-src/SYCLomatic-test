// ====------ BlasUtils_api_test28.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_herk

int main() {
  cublasHandle_t handle;

  float2 *alpha;
  float *beta;
  float2 *a;
  float2 *b;
  float2 *c;

  cublasCherkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, 2, 3, alpha, a, 3, b, 3, beta, c, 2);
  return 0;
}
