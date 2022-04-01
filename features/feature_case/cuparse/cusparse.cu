// ====------ cusparse.cu---------- *- CUDA -* ----===////
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
float alpha;
const float* csrValA;
const int* csrRowPtrA;
const int* csrColIndA;
const float* x;
float beta;
float* y;
cusparseHandle_t handle;
cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
cusparseMatDescr_t descrA;

int main(){
  cusparseMatDescr_t descr1 = 0, descr2 = 0;
  cusparseMatDescr_t descr3 = 0;
  cudaStream_t s;

  cusparsePointerMode_t mode = CUSPARSE_POINTER_MODE_HOST;
  cusparseGetPointerMode(handle, &mode);
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);

  constexpr int zero = 0;
  cusparseDiagType_t diag0 = CUSPARSE_DIAG_TYPE_NON_UNIT;
  cusparseFillMode_t fill0 = CUSPARSE_FILL_MODE_LOWER;
  cusparseIndexBase_t base0 = CUSPARSE_INDEX_BASE_ZERO;
  cusparseMatrixType_t type0 = CUSPARSE_MATRIX_TYPE_GENERAL;
  cusparseSetMatDiagType(descrA, (cusparseDiagType_t)zero);
  cusparseSetMatFillMode(descrA, (cusparseFillMode_t)zero);
  cusparseSetMatIndexBase(descrA, (cusparseIndexBase_t)zero);
  cusparseSetMatType(descrA, (cusparseMatrixType_t)zero);
  diag0 = cusparseGetMatDiagType(descrA);
  fill0 = cusparseGetMatFillMode(descrA);
  base0 = cusparseGetMatIndexBase(descrA);
  type0 = cusparseGetMatType(descrA);

  cusparseCreate(&handle);
  cusparseSetStream(handle,s);
  cusparseGetStream(handle,&s);

  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseScsrmv(handle, (cusparseOperation_t)zero, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);

  cuComplex alpha_C, beta_C, *csrValA_C, *x_C, *y_C;

  cusparseCcsrmv(handle, transA, m, n, nnz, &alpha_C, descrA, csrValA_C, csrRowPtrA, csrColIndA, x_C, &beta_C, y_C);

  cusparseScsrmm(handle, transA, m, n, k, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, ldb, &beta, y, ldc);

  cusparseCcsrmm(handle, transA, m, n, k, nnz, &alpha_C, descrA, csrValA_C, csrRowPtrA, csrColIndA, x_C, ldb, &beta_C, y_C, ldc);

  cusparseStatus_t status;

  if(status = cusparseScsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y)){}

  for(status = cusparseScsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);;){}

  switch(status = cusparseScsrmv(handle, transA, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y)){}

  cusparseSolveAnalysisInfo_t info;
  cusparseCreateSolveAnalysisInfo(&info);
  cusparseScsrsv_analysis(handle, transA, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info);
  cusparseDestroySolveAnalysisInfo(info);

  cusparseCcsrsv_analysis(handle, transA, m, nnz, descrA, csrValA_C, csrRowPtrA, csrColIndA, info);

  cusparseDestroyMatDescr(descrA);
  cusparseDestroy(handle);
}

int foo(cusparseMatDescr_t descrB){
  return cusparseScsrmv(handle, transA, m, n, nnz, &alpha, descrB, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);
}

void foo2(cusparseMatDescr_t descrB){
  cusparseScsrmv(handle, transA, m, n, nnz, &alpha, descrB, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);
}