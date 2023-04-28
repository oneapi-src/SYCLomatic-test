// ===------ LapackUtils_api_test22.cu -------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //


// TEST_FEATURE: LapackUtils_syhegvd

#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t handle;
  cusolverEigType_t itype;
  cusolverEigMode_t jobz;
  cublasFillMode_t uplo;
  int n;
  float *A;
  int lda;
  float *B;
  int ldb;
  float *W;
  float *work;
  int lwork;
  int *info;
  syevjInfo_t params;

  cusolverDnSsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork,
                   info, params);
  return 0;
}
