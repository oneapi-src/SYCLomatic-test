// ====------ cublasLegacyCZ.cu---------- *- CUDA -* ----===////
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

int main() {

  cublasStatus status;
  int n = 275;
  int m = 275;
  int k = 275;
  int lda = 275;
  int ldb = 275;
  int ldc = 275;
  const cuComplex *A_C = 0;
  const cuComplex *B_C = 0;
  cuComplex *C_C = 0;
  cuComplex alpha_C = make_cuComplex(1,0);
  cuComplex beta_C = make_cuComplex(0,0);
  const cuDoubleComplex *A_Z = 0;
  const cuDoubleComplex *B_Z = 0;
  cuDoubleComplex *C_Z = 0;
  cuDoubleComplex alpha_Z = make_cuDoubleComplex(1,0);
  cuDoubleComplex beta_Z = make_cuDoubleComplex(0,0);

  cuComplex *x_C = 0;
  cuDoubleComplex *x_Z = 0;
  cuComplex *y_C = 0;
  cuDoubleComplex *y_Z = 0;
  int incx = 1;
  int incy = 1;
  int *result = 0;
  cuComplex *result_C = 0;
  cuDoubleComplex *result_Z = 0;
  float *result_S = 0;
  double *result_D = 0;

  float *x_f = 0;
  float *y_f = 0;
  double *x_d = 0;
  double *y_d = 0;
  float *x_S = 0;
  float *y_S = 0;
  double *x_D = 0;
  double *y_D = 0;

  float alpha_S = 0;
  double alpha_D = 0;
  float beta_S = 0;
  double beta_D = 0;
  //level1

  //cublasI<t>amax
  int res = cublasIcamax(n, x_C, incx);

  *result = cublasIzamax(n, x_Z, incx);

  //cublasI<t>amin
  *result = cublasIcamin(n, x_C, incx);

  *result = cublasIzamin(n, x_Z, incx);

  //cublas<t>asum
  *result_S = cublasScasum(n, x_C, incx);

  *result_D = cublasDzasum(n, x_Z, incx);

  //cublas<t>dot
  cuComplex resCuComplex = cublasCdotu(n, x_C, incx, y_C, incy);

  *result_C = cublasCdotc(n, x_C, incx, y_C, incy);

  *result_Z = cublasZdotu(n, x_Z, incx, y_Z, incy);

  *result_Z = cublasZdotc(n, x_Z, incx, y_Z, incy);

  //cublas<t>nrm2
  *result_S = cublasScnrm2(n, x_C, incx);

  *result_D = cublasDznrm2(n, x_Z, incx);




  //cublas<t>axpy
  cublasCaxpy(n, alpha_C, x_C, incx, result_C, incy);

  cublasZaxpy(n, alpha_Z, x_Z, incx, result_Z, incy);

  //cublas<t>copy
  cublasCcopy(n, x_C, incx, result_C, incy);

  cublasZcopy(n, x_Z, incx, result_Z, incy);


  //cublas<t>rot
  cublasCsrot(n, x_C, incx, y_C, incy, *x_S, *y_S);

  cublasZdrot(n, x_Z, incx, y_Z, incy, *x_D, *y_D);


  //cublas<t>scal
  cublasCscal(n, alpha_C, x_C, incx);

  cublasZscal(n, alpha_Z, x_Z, incx);

  cublasCsscal(n, alpha_S, x_C, incx);

  cublasZdscal(n, alpha_D, x_Z, incx);

  //cublas<t>swap
  cublasCswap(n, x_C, incx, y_C, incy);

  cublasZswap(n, x_Z, incx, y_Z, incy);

  //level2
  //cublas<t>gbmv
  cublasCgbmv('N', m, n, m, n, alpha_C, x_C, lda, y_C, incx, beta_C, result_C, incy);

  cublasZgbmv( 'N', m, n, m, n, alpha_Z, x_Z, lda, y_Z, incx, beta_Z, result_Z, incy);

  //cublas<t>gemv
  cublasCgemv('N', m, n, alpha_C, x_C, lda, y_C, incx, beta_C, result_C, incy);

  cublasZgemv('N', m, n, alpha_Z, x_Z, lda, y_Z, incx, beta_Z, result_Z, incy);

  //cublas<t>ger
  cublasCgeru(m, n, alpha_C, x_C, incx, y_C, incy, result_C, lda);

  cublasCgerc(m, n, alpha_C, x_C, incx, y_C, incy, result_C, lda);

  cublasZgeru(m, n, alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);

  cublasZgerc(m, n, alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);



  //cublas<t>tbmv
  cublasCtbmv('U', 'N', 'U', n, n, x_C, lda, result_C, incy);

  cublasZtbmv('u', 'N', 'u', n, n, x_Z, lda, result_Z, incy);

  //cublas<t>tbsv
  cublasCtbsv('L', 'N', 'U', n, n, x_C, lda, result_C, incy);

  cublasZtbsv('l', 'N', 'U', n, n, x_Z, lda, result_Z, incy);

  //cublas<t>tpmv
  cublasCtpmv('U', 'N', 'U', n, x_C, result_C, incy);

  cublasZtpmv('U', 'N', 'U', n, x_Z, result_Z, incy);

  //cublas<t>tpsv
  cublasCtpsv('U', 'N', 'U', n, x_C, result_C, incy);

  cublasZtpsv('U', 'N', 'U', n, x_Z, result_Z, incy);

  //cublas<t>trmv
  cublasCtrmv('U', 'N', 'U', n, x_C, lda, result_C, incy);

  cublasZtrmv('U', 'N', 'U', n, x_Z, lda, result_Z, incy);

  //cublas<t>trsv
  cublasCtrsv('U', 'N', 'U', n, x_C, lda, result_C, incy);


  cublasZtrsv('U', 'N', 'U', n, x_Z, lda, result_Z, incy);

  //chemv
  cublasChemv ('U', n, alpha_C, A_C, lda, x_C, incx, beta_C, y_C, incy);

  cublasZhemv ('U', n, alpha_Z, A_Z, lda, x_Z, incx, beta_Z, y_Z, incy);

  cublasChbmv ('U', n, k, alpha_C, A_C, lda, x_C, incx, beta_C, y_C, incy);

  cublasZhbmv ('U', n, k, alpha_Z, A_Z, lda, x_Z, incx, beta_Z, y_Z, incy);

  cublasChpmv('U', n, alpha_C, A_C, x_C, incx, beta_C, y_C, incy);

  cublasZhpmv('U', n, alpha_Z, A_Z, x_Z, incx, beta_Z, y_Z, incy);

  cublasCher ('U', n, alpha_S, x_C, incx, C_C, lda);

  cublasZher ('U', n, alpha_D, x_Z, incx, C_Z, lda);

  cublasCher2 ('U', n, alpha_C, x_C, incx, y_C, incy, C_C, lda);

  cublasZher2 ('U', n, alpha_Z, x_Z, incx, y_Z, incy, C_Z, lda);

  cublasChpr ('U', n, alpha_S, x_C, incx, C_C);

  cublasZhpr ('U', n, alpha_D, x_Z, incx, C_Z);

  cublasChpr2 ('U', n, alpha_C, x_C, incx, y_C, incy, C_C);

  cublasZhpr2 ('U', n, alpha_Z, x_Z, incx, y_Z, incy, C_Z);


  //level3
  cublasCgemm('N', 'N', m, n, k, alpha_C, A_C, lda, B_C, ldb, beta_C, C_C, ldc);

  cublasZgemm('N', 'N', m, n, k, alpha_Z, A_Z, lda, B_Z, ldb, beta_Z, C_Z, ldc);

  // cublas<T>symm
  cublasCsymm('R', 'L', m, n, alpha_C, A_C, lda, B_C, ldb, beta_C, C_C, ldc);

  cublasZsymm('r', 'L', m, n, alpha_Z, A_Z, lda, B_Z, ldb, beta_Z, C_Z, ldc);

  cublasCsyrk('U', 'T', n, k, alpha_C, A_C, lda, beta_C, C_C, ldc);

  cublasZsyrk('U', 't', n, k, alpha_Z, A_Z, lda, beta_Z, C_Z, ldc);

  cublasCherk('U', 't', n, k, alpha_S, A_C, lda, beta_S, C_C, ldc);

  cublasZherk('U', 't', n, k, alpha_D, A_Z, lda, beta_D, C_Z, ldc);

  // cublas<T>syr2k
  cublasCsyr2k('U', 'C', n, k, alpha_C, A_C, lda, B_C, ldb, beta_C, C_C, ldc);

  cublasZsyr2k('U', 'c', n, k, alpha_Z, A_Z, lda, B_Z, ldb, beta_Z, C_Z, ldc);

  cublasCher2k('U', 'c', n, k, alpha_C, A_C, lda, B_C, ldb, beta_S, C_C, ldc);

  cublasZher2k('U', 'c', n, k, alpha_Z, A_Z, lda, B_Z, ldb, beta_D, C_Z, ldc);

  cublasChemm ('R', 'U', m, n, alpha_C, A_C, lda, B_C, ldb, beta_C, C_C, ldc);

  cublasZhemm ('R', 'U', m, n, alpha_Z, A_Z, lda, B_Z, ldb, beta_Z, C_Z, ldc);

  // cublas<T>trsm
  cublasCtrsm('L', 'U', 'N', 'n', m, n, alpha_C, A_C, lda, C_C, ldc);

  cublasZtrsm('l', 'U', 'N', 'N', m, n, alpha_Z, A_Z, lda, C_Z, ldc);

  cublasZtrsm(foo(), foo(), foo(), foo(), m, n, alpha_Z, A_Z, lda, C_Z, ldc);
}
