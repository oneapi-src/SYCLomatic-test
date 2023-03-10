// ===------ LapackUtils_api_test10.cu -------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

// TEST_FEATURE: LapackUtils_gesvd_scratchpad_size

#include "cusolverDn.h"

int main() {
  float* a_s;
  float* s_s;
  float* u_s;
  float* vt_s;
  cusolverDnHandle_t handle;
  size_t device_ws_size_s;
  size_t host_ws_size_s;
  cusolverDnParams_t params;

  cusolverDnXgesvd_bufferSize(handle, params, 'A', 'A', 2, 2, CUDA_R_32F, a_s, 2, CUDA_R_32F, s_s, CUDA_R_32F, u_s, 2, CUDA_R_32F, vt_s, 2, CUDA_R_32F, &device_ws_size_s, &host_ws_size_s);
  return 0;
}
