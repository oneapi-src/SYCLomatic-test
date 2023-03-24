// ===------ LapackUtils_api_test14.cu -------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

// TEST_FEATURE: LapackUtils_potrs

#include "cusolverDn.h"

int main() {
  float* a_s;
  float* b_s;
  cusolverDnHandle_t handle;
  size_t device_ws_size_s;
  size_t host_ws_size_s;
  cusolverDnParams_t params;
  void* device_ws_s;
  void* host_ws_s;
  int *info;

  cusolverDnXpotrs(handle, params, CUBLAS_FILL_MODE_LOWER, 3, 1, CUDA_R_32F, a_s, 3, CUDA_R_32F, b_s, 3, info);
  return 0;
}
