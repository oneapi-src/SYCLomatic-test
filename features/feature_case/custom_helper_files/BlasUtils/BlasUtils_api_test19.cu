// ====------ BlasUtils_api_test19.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_rot

int main() {
  cublasHandle_t handle;
  void *x;
  void *y;
  void *sin;
  void *cos;

  cublasRotEx(handle, 4, x, CUDA_R_32F, 1,  y, CUDA_R_32F, 1,  cos, sin, CUDA_R_32F, CUDA_R_32F);
  return 0;
}
