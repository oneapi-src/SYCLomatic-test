// ====------ cusparse-usm.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cstdio>
#include <cusparse_v2.h>
#include <cuda_runtime.h>

int m, n, nnz, k, ldb, ldc;
double alpha;
const double* csrValA;
const int* csrRowPtrA;
const int* csrColIndA;
const double* x;
double beta;
double* y;
cusparseHandle_t handle;
cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
cusparseMatDescr_t descrA;

int foo(int aaaaa){
  cusparseMatDescr_t descr1 = 0, descr2 = 0;
  cusparseMatDescr_t descr3 = 0;

  cusparsePointerMode_t mode = CUSPARSE_POINTER_MODE_DEVICE;
  cusparseGetPointerMode(handle, &mode);
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

  cusparseDiagType_t diag0 = CUSPARSE_DIAG_TYPE_NON_UNIT;
  cusparseFillMode_t fill0 = CUSPARSE_FILL_MODE_LOWER;
  cusparseIndexBase_t base0 = CUSPARSE_INDEX_BASE_ZERO;
  cusparseMatrixType_t type0 = CUSPARSE_MATRIX_TYPE_GENERAL;
  cusparseSetMatDiagType(descrA, (cusparseDiagType_t)aaaaa);
  cusparseSetMatFillMode(descrA, (cusparseFillMode_t)aaaaa);
  cusparseSetMatIndexBase(descrA, (cusparseIndexBase_t)aaaaa);
  cusparseSetMatType(descrA, (cusparseMatrixType_t)aaaaa);
  diag0 = cusparseGetMatDiagType(descrA);
  fill0 = cusparseGetMatFillMode(descrA);
  base0 = cusparseGetMatIndexBase(descrA);
  type0 = cusparseGetMatType(descrA);

  cusparseCreate(&handle);
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, (cusparseMatrixType_t)aaaaa);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseDcsrmv(handle, (cusparseOperation_t)aaaaa, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);

  cuDoubleComplex alpha_Z, beta_Z, *csrValA_Z, *x_Z, *y_Z;

  cusparseZcsrmv(handle, transA, m, n, nnz, &alpha_Z, descrA, csrValA_Z, csrRowPtrA, csrColIndA, x_Z, &beta_Z, y_Z);

  cusparseDcsrmm(handle, transA, m, n, k, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, ldb, &beta, y, ldc);

  cusparseZcsrmm(handle, transA, m, n, k, nnz, &alpha_Z, descrA, csrValA_Z, csrRowPtrA, csrColIndA, x_Z, ldb, &beta_Z, y_Z, ldc);

  cusparseStatus_t status;

  if(status = cusparseDcsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y)){}

  for(status = cusparseDcsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);;){}

  switch(status = cusparseDcsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y)){}

  cusparseSolveAnalysisInfo_t info;
  cusparseCreateSolveAnalysisInfo(&info);
  cusparseDcsrsv_analysis(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info);
  cusparseDestroySolveAnalysisInfo(info);

  cusparseZcsrsv_analysis(handle, transA, m, nnz, descrA, csrValA_Z, csrRowPtrA, csrColIndA, info);

  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);
}

int foo(cusparseMatDescr_t descrB){
  return cusparseDcsrmv(handle, transA, m, n, nnz, &alpha, descrB, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);
}