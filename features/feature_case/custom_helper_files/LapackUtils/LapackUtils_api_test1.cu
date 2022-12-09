// ===------ LapackUtils_api_test1.cu --------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

// TEST_FEATURE: LapackUtils_sygvd

#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t handle;
  float *a_s, *b_s, *w_s, *work_s;
  int lwork_s;
  int *devInfo;
  cusolverDnSsygvd(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_s, 3, b_s, 3, w_s, work_s, lwork_s, devInfo);
  return 0;
}
