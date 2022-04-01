// ====------ BlasUtils_api_test21.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_syrk

int main() {
  cublasHandle_t handle;
  float *alpha;
  float *beta;
  float *a;
  float *b;
  float *c;

  cublasSsyrkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, 2, 3, alpha, a, 3, b, 3, beta, c, 2);
  return 0;
}
