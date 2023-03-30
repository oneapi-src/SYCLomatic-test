// ===------ SparseUtils_api_test5.cu --------------------- *- CUDA -* ---=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

// TEST_FEATURE: SparseUtils_optimize_csrsv

#include "cusparse.h"

int main() {
  cusparseHandle_t handle;
  cusparseOperation_t trans;
  int m;
  int nnz;
  cusparseMatDescr_t descr;
  float *csrVal;
  int *csrRowPtr;
  int *csrColInd;
  cusparseSolveAnalysisInfo_t info;
  cusparseScsrsv_analysis(handle, trans, m, nnz, descr, csrVal, csrRowPtr,
                          csrColInd, info);
  return 0;
}
