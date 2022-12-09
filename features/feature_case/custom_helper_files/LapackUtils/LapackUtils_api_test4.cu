// ===------ LapackUtils_api_test4.cu --------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

// TEST_FEATURE: LapackUtils_potrs_batch

#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t handle;
  float **a_s_ptrs, **b_s_ptrs;
  int *infoArray;
  cusolverDnSpotrsBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, 1, a_s_ptrs, 3, b_s_ptrs, 3, infoArray, 2);
  return 0;
}
