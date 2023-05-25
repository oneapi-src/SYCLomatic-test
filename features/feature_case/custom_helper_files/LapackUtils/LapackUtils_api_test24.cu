// ===------ LapackUtils_api_test24.cu -------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //


// TEST_FEATURE: LapackUtils_syheev_scratchpad_size

#include "cusolverDn.h"

int main() {
  float *a_s;
  float *w_s;
  cusolverDnHandle_t handle;
  syevjInfo_t params;
  int lwork_s;
  cusolverDnSsyevj_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_s, 2, w_s, &lwork_s, params);
  return 0;
}
