// ====------ cublasIsamax_etc.cu---------- *- CUDA -* ----===////
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

int foo();

int main() {
  cublasStatus_t status;
  cublasHandle_t handle;
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
  //level1
  //cublasI<t>amax
  status = cublasIsamax(handle, n, x_S, incx, result);
  cublasIsamax(handle, n, x_S, incx, result);

  status = cublasIdamax(handle, n, x_D, incx, result);
  cublasIdamax(handle, n, x_D, incx, result);

  //cublasI<t>amin
  status = cublasIsamin(handle, n, x_S, incx, result);
  cublasIsamin(handle, n, x_S, incx, result);

  status = cublasIdamin(handle, n, x_D, incx, result);
  cublasIdamin(handle, n, x_D, incx, result);

  //cublas<t>asum
  status = cublasSasum(handle, n, x_S, incx, result_S);
  cublasSasum(handle, n, x_S, incx, result_S);

  status = cublasDasum(handle, n, x_D, incx, result_D);
  cublasDasum(handle, n, x_D, incx, result_D);

  //cublas<t>axpy
  status = cublasSaxpy(handle, n, &alpha_S, x_S, incx, result_S, incy);
  cublasSaxpy(handle, n, &alpha_S, x_S, incx, result_S, incy);

  status = cublasDaxpy(handle, n, &alpha_D, x_D, incx, result_D, incy);
  cublasDaxpy(handle, n, &alpha_D, x_D, incx, result_D, incy);

  //cublas<t>copy
  status = cublasScopy(handle, n, x_S, incx, result_S, incy);
  cublasScopy(handle, n, x_S, incx, result_S, incy);

  status = cublasDcopy(handle, n, x_D, incx, result_D, incy);
  cublasDcopy(handle, n, x_D, incx, result_D, incy);

  //cublas<t>dot
  status = cublasSdot(handle, n, x_S, incx, y_S, incy, result_S);
  cublasSdot(handle, n, x_S, incx, y_S, incy, result_S);

  status = cublasDdot(handle, n, x_D, incx, y_D, incy, result_D);
  cublasDdot(handle, n, x_D, incx, y_D, incy, result_D);

  //cublas<t>nrm2
  status = cublasSnrm2(handle, n, x_S, incx, result_S);
  cublasSnrm2(handle, n, x_S, incx, result_S);

  status = cublasDnrm2(handle, n, x_D, incx, result_D);
  cublasDnrm2(handle, n, x_D, incx, result_D);

  float *x_f = 0;
  float *y_f = 0;
  double *x_d = 0;
  double *y_d = 0;
  //cublas<t>rot
  status = cublasSrot(handle, n, x_f, incx, y_f, incy, x_S, y_S);
  cublasSrot(handle, n, x_f, incx, y_f, incy, x_S, y_S);

  status = cublasDrot(handle, n, x_d, incx, y_d, incy, x_D, y_D);
  cublasDrot(handle, n, x_d, incx, y_d, incy, x_D, y_D);

  //cublas<t>rotg
  status = cublasSrotg(handle, x_f, y_f, x_f, y_f);
  cublasSrotg(handle, x_f, y_f, x_f, y_f);

  status = cublasDrotg(handle, x_d, y_d, x_d, y_d);
  cublasDrotg(handle, x_d, y_d, x_d, y_d);

  //cublas<t>rotm
  status = cublasSrotm(handle, n, x_f, incx, y_f, incy, x_S);
  cublasSrotm(handle, n, x_f, incx, y_f, incy, x_S);

  status = cublasDrotm(handle, n, x_d, incx, y_d, incy, x_D);
  cublasDrotm(handle, n, x_d, incx, y_d, incy, x_D);

  //cublas<t>rotmg
  status = cublasSrotmg(handle, x_f, y_f, y_f, x_S, y_f);
  cublasSrotmg(handle, x_f, y_f, y_f, x_S, y_f);

  status = cublasDrotmg(handle, x_d, y_d, y_d, x_D, y_d);
  cublasDrotmg(handle, x_d, y_d, y_d, x_D, y_d);

  //cublas<t>scal
  status = cublasSscal(handle, n, &alpha_S, x_f, incx);
  cublasSscal(handle, n, &alpha_S, x_f, incx);

  status = cublasDscal(handle, n, &alpha_D, x_d, incx);
  cublasDscal(handle, n, &alpha_D, x_d, incx);

  //cublas<t>swap
  status = cublasSswap(handle, n, x_f, incx, y_f, incy);
  cublasSswap(handle, n, x_f, incx, y_f, incy);

  status = cublasDswap(handle, n, x_d, incx, y_d, incy);
  cublasDswap(handle, n, x_d, incx, y_d, incy);

  int trans0 = 0;
  int trans1 = 1;
  int trans2 = 2;
  //level2
  //cublas<t>gbmv
  status = cublasSgbmv(handle, (cublasOperation_t)trans0, m, n, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSgbmv(handle, CUBLAS_OP_N, m, n, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  status = cublasDgbmv(handle, (cublasOperation_t)trans1, m, n, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDgbmv(handle, CUBLAS_OP_N, m, n, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>gemv
  status = cublasSgemv(handle, (cublasOperation_t)trans2, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  status = cublasDgemv(handle, (cublasOperation_t)0, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>ger
  status = cublasSger(handle, m, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);
  cublasSger(handle, m, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);

  status = cublasDger(handle, m, n, &alpha_D, x_D, incx, y_D, incy, result_D, lda);
  cublasDger(handle, m, n, &alpha_D, x_D, incx, y_D, incy, result_D, lda);

  int fill0 = 0;
  int fill1 = 1;
  //cublas<t>sbmv
  status = cublasSsbmv(handle, (cublasFillMode_t)fill0, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSsbmv(handle, CUBLAS_FILL_MODE_UPPER, m, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  status = cublasDsbmv(handle, (cublasFillMode_t)fill1, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDsbmv(handle, CUBLAS_FILL_MODE_UPPER, m, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>spmv
  status = cublasSspmv(handle, (cublasFillMode_t)0, n, &alpha_S, x_S, y_S, incx, &beta_S, result_S, incy);
  cublasSspmv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, y_S, incx, &beta_S, result_S, incy);

  status = cublasDspmv(handle, (cublasFillMode_t)1, n, &alpha_D, x_D, y_D, incx, &beta_D, result_D, incy);
  cublasDspmv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>spr
  status = cublasSspr(handle, (cublasFillMode_t)fill0, n, &alpha_S, x_S, incx, result_S);
  cublasSspr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, result_S);

  status = cublasDspr(handle, (cublasFillMode_t)fill0, n, &alpha_D, x_D, incx, result_D);
  cublasDspr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, result_D);

  //cublas<t>spr2
  status = cublasSspr2(handle, (cublasFillMode_t)fill0, n, &alpha_S, x_S, incx, y_S, incy, result_S);
  cublasSspr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, y_S, incy, result_S);

  status = cublasDspr2(handle, (cublasFillMode_t)fill0, n, &alpha_D, x_D, incx, y_D, incy, result_D);
  cublasDspr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, y_D, incy, result_D);

  //cublas<t>symv
  status = cublasSsymv(handle, (cublasFillMode_t)fill0, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);

  status = cublasDsymv(handle, (cublasFillMode_t)fill0, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  cublasDsymv(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);

  //cublas<t>syr
  status = cublasSsyr(handle, (cublasFillMode_t)fill0, n, &alpha_S, x_S, incx, result_S, lda);
  cublasSsyr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, result_S, lda);

  status = cublasDsyr(handle, (cublasFillMode_t)fill0, n, &alpha_D, x_D, incx, result_D, lda);
  cublasDsyr(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, result_D, lda);

  //cublas<t>syr2
  status = cublasSsyr2(handle, (cublasFillMode_t)fill0, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);
  cublasSsyr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_S, x_S, incx, y_S, incy, result_S, lda);

  status = cublasDsyr2(handle, (cublasFillMode_t)fill0, n, &alpha_D, x_D, incx, y_D, incy, result_D, lda);
  cublasDsyr2(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha_D, x_D, incx, y_D, incy, result_D, lda);

  int diag0 = 0;
  int diag1 = 1;
  //cublas<t>tbmv
  status = cublasStbmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)1, (cublasDiagType_t)diag0, n, n, x_S, lda, result_S, incy);
  cublasStbmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_S, lda, result_S, incy);

  status = cublasDtbmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)2, (cublasDiagType_t)diag1, n, n, x_D, lda, result_D, incy);
  cublasDtbmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_D, lda, result_D, incy);

  //cublas<t>tbsv
  status = cublasStbsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)0, n, n, x_S, lda, result_S, incy);
  cublasStbsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_S, lda, result_S, incy);

  status = cublasDtbsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)1, n, n, x_D, lda, result_D, incy);
  cublasDtbsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, x_D, lda, result_D, incy);

  //cublas<t>tpmv
  status = cublasStpmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_S, result_S, incy);
  cublasStpmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, result_S, incy);

  status = cublasDtpmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_D, result_D, incy);
  cublasDtpmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, result_D, incy);

  //cublas<t>tpsv
  status = cublasStpsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_S, result_S, incy);
  cublasStpsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, result_S, incy);

  status = cublasDtpsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_D, result_D, incy);
  cublasDtpsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, result_D, incy);

  //cublas<t>trmv
  status = cublasStrmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_S, lda, result_S, incy);
  cublasStrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, lda, result_S, incy);

  status = cublasDtrmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_D, lda, result_D, incy);
  cublasDtrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, lda, result_D, incy);

  //cublas<t>trsv
  status = cublasStrsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_S, lda, result_S, incy);
  cublasStrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_S, lda, result_S, incy);

  status = cublasDtrsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_D, lda, result_D, incy);
  cublasDtrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, x_D, lda, result_D, incy);

  //level3
  int side0 = 0;
  int side1 = 1;
  // cublas<T>gemmStridedBatched
  status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &alpha_S, A_S, n, 16, B_S, n, 16, &beta_S, C_S, n, 16, 10);
  cublasSgemmStridedBatched(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, n, n, n, &alpha_S, A_S, n, 16, B_S, n, 16, &beta_S, C_S, n, 16, 10);

  status = cublasDgemmStridedBatched(handle, CUBLAS_OP_C, CUBLAS_OP_C, n, n, n, &alpha_D, A_D, n, 16, B_D, n, 16, &beta_D, C_D, n, 16, 10);
  cublasDgemmStridedBatched(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, n, n, n, &alpha_D, A_D, n, 16, B_D, n, 16, &beta_D, C_D, n, 16, 10);

  const float** A_S_array;
  const float** B_S_array;
  float** C_S_array;
  const double** A_D_array;
  const double** B_D_array;
  double** C_D_array;
  cublasOperation_t trans3 = CUBLAS_OP_N;

  // cublas<T>symm
  status = cublasSsymm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, m, n, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);
  cublasSsymm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, m, n, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);

  status = cublasDsymm(handle, (cublasSideMode_t)side1, (cublasFillMode_t)fill0, m, n, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);
  cublasDsymm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, m, n, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);

  // cublas<T>syrk
  status = cublasSsyrk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_S, A_S, lda, &beta_S, C_S, ldc);
  cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_S, A_S, lda, &beta_S, C_S, ldc);

  status = cublasDsyrk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_D, A_D, lda, &beta_D, C_D, ldc);
  cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_D, A_D, lda, &beta_D, C_D, ldc);

  // cublas<T>syr2k
  status = cublasSsyr2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);
  cublasSsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_S, A_S, lda, B_S, ldb, &beta_S, C_S, ldc);

  status = cublasDsyr2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);
  cublasDsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alpha_D, A_D, lda, B_D, ldb, &beta_D, C_D, ldc);

  // cublas<T>trsm
  status = cublasStrsm(handle, (cublasSideMode_t)0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, m, n, &alpha_S, A_S, lda, C_S, ldc);
  cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_S, A_S, lda, C_S, ldc);

  status = cublasDtrsm(handle, (cublasSideMode_t)1, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, m, n, &alpha_D, A_D, lda, C_D, ldc);
  cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_D, A_D, lda, C_D, ldc);


  cublasDtrsm(handle, (cublasSideMode_t)foo(), (cublasFillMode_t)foo(), (cublasOperation_t)foo(), (cublasDiagType_t)foo(), m, n, &alpha_D, A_D, lda, C_D, ldc);
  return 0;
}
