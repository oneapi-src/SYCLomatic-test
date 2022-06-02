// ====------ BlasUtils_api_test32.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_trsm_batch

int main() {
  cublasHandle_t handle;
  float * alpha;
  float **a, **b;

  cublasStrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 4, 4, alpha, a, 4, b, 4, 2);
  return 0;
}
