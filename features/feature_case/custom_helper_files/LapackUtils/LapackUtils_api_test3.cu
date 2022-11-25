// ===------ LapackUtils_api_test3.cu --------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

// TEST_FEATURE: LapackUtils_potrf_batch

#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t handle;
  float ** a_s_ptrs;
  int *infoArray;
  cusolverDnSpotrfBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, a_s_ptrs, 3, infoArray, 2);
  return 0;
}
