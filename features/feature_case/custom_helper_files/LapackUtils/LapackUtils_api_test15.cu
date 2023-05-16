// ===------ LapackUtils_api_test15.cu -------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

// TEST_FEATURE: LapackUtils_syheevx_scratchpad_size_T

#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t handle;
  cusolverEigMode_t jobz;
  cusolverEigRange_t range;
  cublasFillMode_t uplo;
  int n;
  const float *A;
  int lda;
  float vl;
  float vu;
  int il;
  int iu;
  int *h_meig;
  const float *W;
  int *lwork;

  cusolverDnSsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il,
                               iu, h_meig, W, lwork);
  return 0;
}
