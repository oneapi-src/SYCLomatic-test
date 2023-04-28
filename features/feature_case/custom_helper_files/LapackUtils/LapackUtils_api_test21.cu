// ===------ LapackUtils_api_test21.cu -------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //


// TEST_FEATURE: LapackUtils_syhegvd_scratchpad_size

#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t handle;
  cusolverEigType_t itype;
  cusolverEigMode_t jobz;
  cublasFillMode_t uplo;
  int n;
  const float *A;
  int lda;
  const float *B;
  int ldb;
  const float *W;
  int *lwork;
  syevjInfo_t params;

  cusolverDnSsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W,
                              lwork, params);
  return 0;
}
