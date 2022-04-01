// ====------ cublas-usm-legacy.cu---------- *- CUDA -* ----===////
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

int n = 275;
int m = 275;
int k = 275;
int lda = 275;
int ldb = 275;
int ldc = 275;

const float *A_S = 0;
const float *B_S = 0;
float *C_S = 0;
float alpha_S = 1.0f;
float beta_S = 0.0f;
const double *A_D = 0;
const double *B_D = 0;
double *C_D = 0;
double alpha_D = 1.0;
double beta_D = 0.0;

const float2 *A_C;
const float2 *B_C;
float2 *C_C;
float2 alpha_C;
float2 beta_C;
const double2 *A_Z;
const double2 *B_Z;
double2 *C_Z;
double2 alpha_Z;
double2 beta_Z;

const float *x_S = 0;
const double *x_D = 0;
const float *y_S = 0;
const double *y_D = 0;
const float2 *x_C;
const float2 *y_C;
const double2 *x_Z;
const double2 *y_Z;

int incx = 1;
int incy = 1;
int *result = 0;
float *result_S = 0;
double *result_D = 0;
float2 *result_C;
double2 *result_Z;

int elemSize = 4;

int main() {

  cublasStatus status = cublasAlloc(n, elemSize, (void **)&C_S);
  cublasAlloc(n, elemSize, (void **)&C_S);

  // level 1

  int res = cublasIsamax(n, x_S, incx);
  res = cublasIdamax(n, x_D, incx);
  res = cublasIcamax(n, x_C, incx);
  res = cublasIzamax(n, x_Z, incx);

  // Because the return value of origin API is the result value, not the status, so keep using lambda here.
  if(cublasIzamax(n, x_Z, incx)){}

  if(0!=cublasIzamax(n, x_Z, incx)){}

  for(cublasCdotc(n, x_C, incx, y_C, incy);;){}

  cublasSrotm(n, result_S, n, result_S, n, x_S);
  cublasDrotm(n, result_D, n, result_D, n, x_D);

  cublasScopy(n, x_S, incx, result_S, incy);
  cublasDcopy(n, x_D, incx, result_D, incy);
  cublasCcopy(n, x_C, incx, result_C, incy);
  cublasZcopy(n, x_Z, incx, result_Z, incy);

  cublasSaxpy(n, alpha_S, x_S, incx, result_S, incy);
  cublasDaxpy(n, alpha_D, x_D, incx, result_D, incy);
  cublasCaxpy(n, alpha_C, x_C, incx, result_C, incy);
  cublasZaxpy(n, alpha_Z, x_Z, incx, result_Z, incy);

  cublasSscal(n, alpha_S, result_S, incx);
  cublasDscal(n, alpha_D, result_D, incx);
  cublasCscal(n, alpha_C, result_C, incx);
  cublasZscal(n, alpha_Z, result_Z, incx);

  *result_S = cublasSnrm2(n, x_S, incx);
  *result_D = cublasDnrm2(n, x_D, incx);
  *result_S = cublasScnrm2(n, x_C, incx);
  *result_D = cublasDznrm2(n, x_Z, incx);

  *result_C = cublasCdotc(n, x_C, incx, y_C, incy);

  *result_Z = cublasZdotu(n, x_Z, incx, y_Z, incy);

  //level 2

  cublasSgemv('N', m, n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);
  cublasDgemv('N', m, n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);
  cublasCgemv('N', m, n, alpha_C, x_C, lda, y_C, incx, beta_C, result_C, incy);
  cublasZgemv('N', m, n, alpha_Z, x_Z, lda, y_Z, incx, beta_Z, result_Z, incy);

  cublasSger(m, n, alpha_S, x_S, incx, y_S, incy, result_S, lda);
  cublasDger(m, n, alpha_D, x_D, incx, y_D, incy, result_D, lda);
  cublasCgeru(m, n, alpha_C, x_C, incx, y_C, incy, result_C, lda);
  cublasCgerc(m, n, alpha_C, x_C, incx, y_C, incy, result_C, lda);
  cublasZgeru(m, n, alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);
  cublasZgerc(m, n, alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);

  //level 3

  cublasSgemm('N', 'N', n, n, n, alpha_S, A_S, n, B_S, n, beta_S, C_S, n);
  cublasDgemm('N', 'N', n, n, n, alpha_D, A_D, n, B_D, n, beta_D, C_D, n);
  cublasCgemm('N', 'N', n, n, n, alpha_C, A_C, n, B_C, n, beta_C, C_C, n);
  cublasZgemm('N', 'N', n, n, n, alpha_Z, A_Z, n, B_Z, n, beta_Z, C_Z, n);

  cublasStrmm('L', 'L', 'N', 'N', n, n, alpha_S, A_S, n, C_S, n);
  cublasDtrmm('L', 'L', 'N', 'N', n, n, alpha_D, A_D, n, C_D, n);
  cublasCtrmm('L', 'L', 'N', 'N', n, n, alpha_C, A_C, n, C_C, n);
  cublasZtrmm('L', 'L', 'N', 'N', n, n, alpha_Z, A_Z, n, C_Z, n);
}

// Because the return value of origin API is the result value, not the status, so keep using lambda here.
int foo(){
  return cublasIzamax(n, x_Z, incx);
}
