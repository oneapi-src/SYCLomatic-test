// ===------ SparseUtils_api_test2.cu --------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

// TEST_FEATURE: SparseUtils_csrmv

#include "cusparse.h"

int main() {
  cusparseHandle_t handle;
  cusparseOperation_t trans;
  int m;
  int n;
  int nnz;
  float *alpha;
  cusparseMatDescr_t descr;
  float *csrVal;
  int *csrRowPtr;
  int *csrColInd;
  float *x;
  float *beta;
  float *y;
  cusparseScsrmv(handle, trans, m, n, nnz, alpha, descr, csrVal, csrRowPtr,
                 csrColInd, x, beta, y);
  return 0;
}
