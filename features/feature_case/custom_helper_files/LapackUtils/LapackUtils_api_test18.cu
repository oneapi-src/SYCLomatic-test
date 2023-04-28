// ===------ LapackUtils_api_test18.cu -------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //


// TEST_FEATURE: LapackUtils_syheevx

#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t handle;
  cusolverDnParams_t params;
  cusolverEigMode_t jobz;
  cusolverEigRange_t range;
  cublasFillMode_t uplo;
  int64_t n;
  cudaDataType dataTypeA;
  void *A;
  int64_t lda;
  void *vl;
  void *vu;
  int64_t il;
  int64_t iu;
  int64_t *meig64;
  cudaDataType dataTypeW;
  void *W;
  cudaDataType computeType;
  void *bufferOnDevice;
  size_t workspaceInBytesOnDevice;
  void *bufferOnHost;
  size_t workspaceInBytesOnHost;
  int *info;

  cusolverDnXsyevdx(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl,
                    vu, il, iu, meig64, dataTypeW, W, computeType,
                    bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost,
                    workspaceInBytesOnHost, info);
  return 0;
}
