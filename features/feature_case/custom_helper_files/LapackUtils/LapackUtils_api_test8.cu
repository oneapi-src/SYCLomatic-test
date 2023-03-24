// ===------ LapackUtils_api_test8.cu --------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

// TEST_FEATURE: LapackUtils_geqrf_scratchpad_size

#include "cusolverDn.h"

int main() {
  float* a_s;
  int64_t* tau_s;
  cusolverDnHandle_t handle;
  size_t device_ws_size_s;
  size_t host_ws_size_s;
  cusolverDnParams_t params;

  cusolverDnXgeqrf_bufferSize(handle, params, 2, 2, CUDA_R_32F, a_s, 2, CUDA_R_32F, tau_s, CUDA_R_32F, &device_ws_size_s, &host_ws_size_s);
  return 0;
}
