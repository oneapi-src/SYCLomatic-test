// ====------ BlasUtils_api_test13.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_dotc

int main() {
  cublasHandle_t handle;
  const void *x;
  const void *y;
  void *res;

  cublasDotcEx(handle, 4, x, CUDA_R_32F, 1, y, CUDA_R_32F, 1, res, CUDA_R_32F, CUDA_R_32F);
  return 0;
}
