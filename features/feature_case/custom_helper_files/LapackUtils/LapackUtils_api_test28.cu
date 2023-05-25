// ===------ LapackUtils_api_test28.cu -------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //


// TEST_FEATURE: LapackUtils_trtri_scratchpad_size

#include "cusolverDn.h"

int main() {
  float *a_s;
  cusolverDnHandle_t handle;
  size_t lwork_s;
  size_t lwork_host_s;
  cusolverDnXtrtri_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, 2, CUDA_R_32F, a_s, 2, &lwork_s, &lwork_host_s);
  return 0;
}
