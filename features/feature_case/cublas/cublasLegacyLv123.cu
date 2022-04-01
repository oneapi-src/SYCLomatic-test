// ====------ cublasLegacyLv123.cu---------- *- CUDA -* ----===////
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

char foo();

cublasStatus status;
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

const float *x_S = 0;
const double *x_D = 0;
const float *y_S = 0;
const double *y_D = 0;
int incx = 1;
int incy = 1;
int *result = 0;
float *result_S = 0;
double *result_D = 0;


float *x_f = 0;
float *y_f = 0;
double *x_d = 0;
double *y_d = 0;

int main() {

  //level1

  //cublasI<t>amax
  int res = cublasIsamax(n, x_S, incx);

  *result = cublasIdamax(n, x_D, incx);

  //cublasI<t>amin
  *result = cublasIsamin(n, x_S, incx);

  *result = cublasIdamin(n, x_D, incx);

  //cublas<t>asum
  *result_S = cublasSasum(n, x_S, incx);

  *result_D = cublasDasum(n, x_D, incx);

  //cublas<t>dot
  *result_S = cublasSdot(n, x_S, incx, y_S, incy);

  *result_D = cublasDdot(n, x_D, incx, y_D, incy);

  //cublas<t>nrm2
  *result_S = cublasSnrm2(n, x_S, incx);

  *result_D = cublasDnrm2(n, x_D, incx);




  //cublas<t>axpy
  cublasSaxpy(n, alpha_S, x_S, incx, result_S, incy);

  cublasDaxpy(n, alpha_D, x_D, incx, result_D, incy);

  //cublas<t>copy
  cublasScopy(n, x_S, incx, result_S, incy);

  cublasDcopy(n, x_D, incx, result_D, incy);


  //cublas<t>rot
  cublasSrot(n, x_f, incx, y_f, incy, *x_S, *y_S);

  cublasDrot(n, x_d, incx, y_d, incy, *x_D, *y_D);

  //cublas<t>rotg
  cublasSrotg(x_f, y_f, x_f, y_f);

  cublasDrotg(x_d, y_d, x_d, y_d);

  //cublas<t>rotm
  cublasSrotm(n, x_f, incx, y_f, incy, x_S);

  cublasDrotm(n, x_d, incx, y_d, incy, x_D);

  //cublas<t>rotmg
  cublasSrotmg(x_f, y_f, y_f, x_S, y_f);

  cublasDrotmg(x_d, y_d, y_d, x_D, y_d);

  //cublas<t>scal
  cublasSscal(n, alpha_S, x_f, incx);

  cublasDscal(n, alpha_D, x_d, incx);

  cublasSswap(n, x_f, incx, y_f, incy);

  cublasDswap(n, x_d, incx, y_d, incy);

  //level2
  //cublas<t>gbmv
  cublasSgbmv('N', m, n, m, n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);

  cublasDgbmv( 'N', m, n, m, n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);

  //cublas<t>gemv
  cublasSgemv('N', m, n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);

  cublasDgemv('N', m, n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);

  //cublas<t>ger
  cublasSger(m, n, alpha_S, x_S, incx, y_S, incy, result_S, lda);

  cublasDger(m, n, alpha_D, x_D, incx, y_D, incy, result_D, lda);

  //cublas<t>sbmv
  cublasSsbmv('U', m, n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);

  cublasDsbmv('U', m, n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);

  //cublas<t>spmv
  cublasSspmv('U', n, alpha_S, x_S, y_S, incx, beta_S, result_S, incy);

  cublasDspmv('U', n, alpha_D, x_D, y_D, incx, beta_D, result_D, incy);

  //cublas<t>spr
  cublasSspr('U', n, alpha_S, x_S, incx, result_S);

  cublasDspr('U', n, alpha_D, x_D, incx, result_D);

  //cublas<t>spr2
  cublasSspr2('U', n, alpha_S, x_S, incx, y_S, incy, result_S);

  cublasDspr2('U', n, alpha_D, x_D, incx, y_D, incy, result_D);

  //cublas<t>symv
  cublasSsymv('U', n, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);

  cublasDsymv('U', n, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy);

  //cublas<t>syr
  cublasSsyr('U', n, alpha_S, x_S, incx, result_S, lda);

  cublasDsyr('U', n, alpha_D, x_D, incx, result_D, lda);

  //cublas<t>syr2
  cublasSsyr2('U', n, alpha_S, x_S, incx, y_S, incy, result_S, lda);

  cublasDsyr2('U', n, alpha_D, x_D, incx, y_D, incy, result_D, lda);

  //cublas<t>tbmv
  cublasStbmv('U', 'N', 'U', n, n, x_S, lda, result_S, incy);

  cublasDtbmv('u', 'N', 'u', n, n, x_D, lda, result_D, incy);

  //cublas<t>tbsv
  cublasStbsv('L', 'N', 'U', n, n, x_S, lda, result_S, incy);

  cublasDtbsv('l', 'N', 'U', n, n, x_D, lda, result_D, incy);

  //cublas<t>tpmv
  cublasStpmv('U', 'N', 'U', n, x_S, result_S, incy);

  cublasDtpmv('U', 'N', 'U', n, x_D, result_D, incy);

  //cublas<t>tpsv
  cublasStpsv('U', 'N', 'U', n, x_S, result_S, incy);

  cublasDtpsv('U', 'N', 'U', n, x_D, result_D, incy);

  //cublas<t>trmv
  cublasStrmv('U', 'N', 'U', n, x_S, lda, result_S, incy);

  cublasDtrmv('U', 'N', 'U', n, x_D, lda, result_D, incy);

  //cublas<t>trsv
  cublasStrsv('U', 'N', 'U', n, x_S, lda, result_S, incy);


  cublasDtrsv('U', 'N', 'U', n, x_D, lda, result_D, incy);

  //level3

  // cublas<T>symm
  cublasSsymm('R', 'L', m, n, alpha_S, A_S, lda, B_S, ldb, beta_S, C_S, ldc);

  cublasDsymm('r', 'L', m, n, alpha_D, A_D, lda, B_D, ldb, beta_D, C_D, ldc);

  cublasSsyrk('U', 'T', n, k, alpha_S, A_S, lda, beta_S, C_S, ldc);

  cublasDsyrk('U', 't', n, k, alpha_D, A_D, lda, beta_D, C_D, ldc);

  // cublas<T>syr2k
  cublasSsyr2k('U', 'C', n, k, alpha_S, A_S, lda, B_S, ldb, beta_S, C_S, ldc);

  cublasDsyr2k('U', 'c', n, k, alpha_D, A_D, lda, B_D, ldb, beta_D, C_D, ldc);

  // cublas<T>trsm
  cublasStrsm('L', 'U', 'N', 'n', m, n, alpha_S, A_S, lda, C_S, ldc);

  cublasDtrsm('l', 'U', 'N', 'N', m, n, alpha_D, A_D, lda, C_D, ldc);

  cublasSgemm('T', 'C', n, n, n, alpha_S, A_S, n, B_S, n, beta_S, C_S, n);

  cublasDgemm('N', 'n', n, n, n, alpha_D, A_D, n, B_D, n, beta_D, C_D, n);

  cublasDtrsm(foo(), foo(), foo(), foo(), m, n, alpha_D, A_D, lda, C_D, ldc);

  for(cublasDgemm('N', 'n', n, n, n, alpha_D, A_D, n, B_D, n, beta_D, C_D, n);;){}

  // Because the return value of origin API is the result value, not the status, so keep using lambda here.
  for(int i = cublasIsamax(n, x_S, incx);;){}

}


// Because the return value of origin API is the result value, not the status, so keep using lambda here.
int bar(){
  return cublasIsamax(n, x_S, incx);
}