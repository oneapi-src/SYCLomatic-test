// ===------ LapackUtils_api_test17.cu -------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //


// TEST_FEATURE: LapackUtils_syheevx_scratchpad_size

#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t handle;
  cusolverDnParams_t params;
  cusolverEigMode_t jobz;
  cusolverEigRange_t range;
  cublasFillMode_t uplo;
  int64_t n;
  cudaDataType dataTypeA;
  const void *A;
  int64_t lda;
  void *vl;
  void *vu;
  int64_t il;
  int64_t iu;
  int64_t *h_meig;
  cudaDataType dataTypeW;
  const void *W;
  cudaDataType computeType;
  size_t *workspaceInBytesOnDevice;
  size_t *workspaceInBytesOnHost;

  cusolverDnXsyevdx_bufferSize(handle, params, jobz, range, uplo, n, dataTypeA,
                               A, lda, vl, vu, il, iu, h_meig, dataTypeW, W,
                               computeType, workspaceInBytesOnDevice,
                               workspaceInBytesOnHost);
  return 0;
}
