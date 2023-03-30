// ===------ SparseUtils_api_test3.cu --------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

// CHECK: 11
// TEST_FEATURE: SparseUtils_csrmm

#include "cusparse.h"

int main() {
  cusparseHandle_t handle;
  cusparseOperation_t trans;
  int m;
  int n;
  int k;
  int nnz;
  float *alpha;
  cusparseMatDescr_t descr;
  float *csrVal;
  int *csrRowPtr;
  int *csrColInd;
  float* B;
  int ldb;
  float* beta;
  float* C;
  int ldc;
  cusparseScsrmm(handle, trans, m, n, k, nnz, alpha, descr, csrVal, csrRowPtr,
                 csrColInd, B, ldb, beta, C, ldc);
  return 0;
}
