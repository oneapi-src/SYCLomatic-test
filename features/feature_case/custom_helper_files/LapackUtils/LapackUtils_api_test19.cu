// ===------ LapackUtils_api_test19.cu -------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //


// TEST_FEATURE: LapackUtils_syhegvx_scratchpad_size

#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t handle;
  cusolverEigType_t itype;
  cusolverEigMode_t jobz;
  cusolverEigRange_t range;
  cublasFillMode_t uplo;
  int n;
  const float *A;
  int lda;
  const float *B;
  int ldb;
  float vl;
  float vu;
  int il;
  int iu;
  int *h_meig;
  const float *W;
  int *lwork;

  cusolverDnSsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B,
                               ldb, vl, vu, il, iu, h_meig, W, lwork);
  return 0;
}
