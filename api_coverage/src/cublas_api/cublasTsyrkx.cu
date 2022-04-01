// ====------ cublasTsyrkx.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//


#include <cstdio>
#include <cublas.h>
#include <cuda_runtime.h>

#define macro_a (cublasOperation_t)1

#define macro_b (cublasFillMode_t)1

cublasFillMode_t foo(){
  return CUBLAS_FILL_MODE_LOWER;
}

cublasOperation_t bar(){
  return CUBLAS_OP_T;
}

int main() {
  int n = 275;
  int k = 275;
  int lda = 1;
  int ldb = 1;
  int ldc = 1;

  float alpha_s = 1;
  float beta_s = 1;

  double alpha_d = 1;
  double beta_d = 1;

  cublasHandle_t handle;
  cublasStatus_t status;

  float* A_s=0;
  float* B_s=0;
  float* C_s=0;

  double* A_d=0;
  double* B_d=0;
  double* C_d=0;

  int trans0 = 0;
  int trans1 = 1;
  int fill0 = 0;
  int fill1 = 1;

  status = cublasSsyrkx(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_s, A_s, lda, B_s, ldb, &beta_s, C_s, ldc);
  cublasSsyrkx(handle, (cublasFillMode_t)fill1, (cublasOperation_t)trans1, n, k, &alpha_s, A_s, lda, B_s, ldb, &beta_s, C_s, ldc);

  status = cublasDsyrkx(handle, (cublasFillMode_t)0, (cublasOperation_t)0, n, k, &alpha_d, A_d, lda, B_d, ldb, &beta_d, C_d, ldc);
  cublasDsyrkx(handle, (cublasFillMode_t)1, (cublasOperation_t)1, n, k, &alpha_d, A_d, lda, B_d, ldb, &beta_d, C_d, ldc);




  cublasDsyrkx(handle, foo(), macro_a, n, k, &alpha_d, A_d, lda, B_d, ldb, &beta_d, C_d, ldc);


  cublasDsyrkx(handle, macro_b, bar(), n, k, &alpha_d, A_d, lda, B_d, ldb, &beta_d, C_d, ldc);


  cublasDsyrkx(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, k, &alpha_d, A_d, lda, B_d, ldb, &beta_d, C_d, ldc);

  return 0;
}
