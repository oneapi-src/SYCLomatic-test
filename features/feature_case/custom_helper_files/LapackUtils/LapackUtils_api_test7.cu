// ===------ LapackUtils_api_test7.cu --------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

// TEST_FEATURE: LapackUtils_getrs

#include "cusolverDn.h"

int main() {
  float* a_s;
  int64_t* ipiv_s;
  float* b_s;
  cusolverDnHandle_t handle;
  cusolverDnParams_t params;
  int *info;

  cusolverDnXgetrs(handle, params, CUBLAS_OP_N, 2, 3, CUDA_R_32F, a_s, 2, ipiv_s, CUDA_R_32F, b_s, 2, info);
  return 0;
}
