// ===------ LapackUtils_api_test13.cu -------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

// TEST_FEATURE: LapackUtils_potrf

#include "cusolverDn.h"

int main() {
  float* a_s;
  cusolverDnHandle_t handle;
  size_t device_ws_size_s;
  size_t host_ws_size_s;
  cusolverDnParams_t params;
  void* device_ws_s;
  void* host_ws_s;
  int *info;

  cusolverDnXpotrf(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_R_32F, a_s, 3, CUDA_R_32F, device_ws_s, device_ws_size_s, host_ws_s, host_ws_size_s, info);
  return 0;
}
