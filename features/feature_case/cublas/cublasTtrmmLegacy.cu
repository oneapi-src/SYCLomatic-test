// ====------ cublasTtrmmLegacy.cu---------- *- CUDA -* ----===////
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

int main(){
  cublasStatus_t status;
  cublasHandle_t handle;
  int n = 275;
  int m = 275;
  int lda = 275;
  int ldb = 275;
  const float *A_S = 0;
  float *B_S = 0;
  float alpha_S = 1.0f;
  const double *A_D = 0;
  double *B_D = 0;
  double alpha_D = 1.0;
  const cuComplex *A_C = 0;
  cuComplex *B_C = 0;
  cuComplex alpha_C = make_cuComplex(1.0f,0.0f);
  const cuDoubleComplex *A_Z = 0;
  cuDoubleComplex *B_Z = 0;
  cuDoubleComplex alpha_Z = make_cuDoubleComplex(1.0,0.0);


  //Legacy
  cublasStrmm('L', 'U', 'N', 'N', m, n, alpha_S, A_S, lda, B_S, ldb);

  cublasDtrmm('L', 'U', 'N', 'N', m, n, alpha_D, A_D, lda, B_D, ldb);

  cublasCtrmm('L', 'U', 'N', 'N', m, n, alpha_C, A_C, lda, B_C, ldb);

  cublasZtrmm('L', 'U', 'N', 'N', m, n, alpha_Z, A_Z, lda, B_Z, ldb);
}
