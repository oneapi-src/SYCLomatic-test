// ====------ cublasGetSetMatrix.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

constexpr int foo(int i) {
  return i;
}

int main() {
  int rowsA = 100;
  int colsA = 100;
  int lda = 100;
  int ldb = 100;
  float *A = NULL;
  float *d_A = NULL;
  cublasStatus_t status;
  cudaStream_t stream;

#define LDA_MARCO 100
  const int ConstLda = 100;
  status = cublasSetMatrix(100, colsA, sizeof(A[0]), A, LDA_MARCO, d_A, 100);

  cublasSetMatrix(100, colsA, sizeof(A[0]), A, ConstLda, d_A, 100);

  cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, lda, d_A, ldb);

#define LDB_MARCO 99
  cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, 100, d_A, LDB_MARCO);

  cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, 100, d_A, 100);

  cublasSetMatrix(99, colsA, sizeof(A[0]), A, 100, d_A, 100);

  status = cublasSetMatrix(99, colsA, sizeof(A[0]), A, 100, d_A, 100);

  const int ConstLdaNE = lda;
  const int ConstLdbNE = ldb;
  cublasGetMatrix(rowsA, colsA, sizeof(A[0]), A, ConstLdaNE, d_A, ConstLdbNE);

  const int ConstLdaT = 100;
  const int ConstLdbT = 100;
  constexpr int ConstExprLda = 101;
  constexpr int ConstExprLdb = 101;
  cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, foo(lda), d_A, foo(ldb));

  cublasSetMatrix(100, colsA, sizeof(A[0]), A, foo(ConstLdaT), d_A, foo(ConstLdbT));

  cublasGetMatrix(100, colsA, sizeof(A[0]), A, foo(ConstExprLda), d_A, ConstExprLdb);

  status = cublasSetMatrixAsync(100, colsA, sizeof(A[0]), A, 100, d_A, 100, stream);
  cublasSetMatrixAsync(100, colsA, sizeof(A[0]), A, 100, d_A, 100, stream);

  status = cublasGetMatrixAsync(100, colsA, sizeof(A[0]), A, 100, d_A, 100, stream);
  cublasGetMatrixAsync(100, colsA, sizeof(A[0]), A, 100, d_A, 100, stream);

  return 0;
}
