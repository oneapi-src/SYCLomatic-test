// ===------ LapackUtils_api_test25.cu -------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //


// TEST_FEATURE: LapackUtils_syheevd

#include "cusolverDn.h"

int main() {
  float *a_s;
  float *w_s;
  cusolverDnHandle_t handle;
  cusolverDnParams_t params;
  size_t lwork_s;
  size_t lwork_host_s;
  void *device_ws_s;
  void *host_ws_s;
  int *info;
  cusolverDnXsyevd(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 2, CUDA_R_32F, a_s, 2, CUDA_R_32F, w_s, CUDA_R_32F, device_ws_s, lwork_s, host_ws_s, lwork_host_s, info);
  return 0;
}
