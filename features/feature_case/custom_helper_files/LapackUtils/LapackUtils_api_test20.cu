// ===------ LapackUtils_api_test20.cu -------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //


// TEST_FEATURE: LapackUtils_syhegvx

#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t handle;
  cusolverEigType_t itype;
  cusolverEigMode_t jobz;
  cusolverEigRange_t range;
  cublasFillMode_t uplo;
  int n;
  float *A;
  int lda;
  float *B;
  int ldb;
  float vl;
  float vu;
  int il;
  int iu;
  int *h_meig;
  float *W;
  float *work;
  int lwork;
  int *devInfo;

  cusolverDnSsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu,
                    il, iu, h_meig, W, work, lwork, devInfo);
  return 0;
}
