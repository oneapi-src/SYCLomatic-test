// ====------ BlasUtils_api_test31.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_scal

int main() {
  cublasHandle_t handle;
  void * alpha;
  void * x;

  cublasScalEx(handle, 4, alpha, CUDA_R_32F, x, CUDA_R_32F, 1, CUDA_R_32F);
  return 0;
}
