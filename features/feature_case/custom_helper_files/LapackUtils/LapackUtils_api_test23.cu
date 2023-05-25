// ===------ LapackUtils_api_test23.cu -------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //


// TEST_FEATURE: LapackUtils_syheev

#include "cusolverDn.h"

int main() {
  float *a_s;
  float *w_s;
  cusolverDnHandle_t handle;
  syevjInfo_t params;
  int lwork_s;
  float *device_ws_s;
  int *info;
  cusolverDnSsyevj(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, a_s, 2, w_s, device_ws_s, lwork_s, info, params);
  return 0;
}
