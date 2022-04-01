// ====------ cublasRegularCZ.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//


#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(){
  cublasStatus_t status;
  cublasHandle_t handle;

  int* result = 0;
  float* result_f = 0;
  double* result_d = 0;
  cuComplex* x_c = 0;
  cuDoubleComplex* x_z = 0;

  int incx = 1;
  int incy = 1;
  int n = 10;

  //level 1
  status = cublasIcamax(handle, n, x_c, incx, result);
  cublasIcamax(handle, n, x_c, incx, result);

  status = cublasIzamax(handle, n, x_z, incx, result);
  cublasIzamax(handle, n, x_z, incx, result);

  status = cublasIcamin(handle, n, x_c, incx, result);
  cublasIcamin(handle, n, x_c, incx, result);

  status = cublasIzamin(handle, n, x_z, incx, result);
  cublasIzamin(handle, n, x_z, incx, result);

  status = cublasScasum(handle, n, x_c, incx, result_f);
  cublasScasum(handle, n, x_c, incx, result_f);

  status = cublasDzasum(handle, n, x_z, incx, result_d);
  cublasDzasum(handle, n, x_z, incx, result_d);

  cuComplex* alpha_c = 0;
  cuComplex* beta_c = 0;
  cuDoubleComplex* alpha_z = 0;
  cuDoubleComplex* beta_z = 0;
  float* alpha_f = 0;
  double* alpha_d = 0;
  cuComplex* y_c = 0;
  cuDoubleComplex* y_z = 0;

  status = cublasCaxpy(handle, n, alpha_c, x_c, incx, y_c, incy);
  cublasCaxpy(handle, n, alpha_c, x_c, incx, y_c, incy);

  status = cublasZaxpy(handle, n, alpha_z, x_z, incx, y_z, incy);
  cublasZaxpy(handle, n, alpha_z, x_z, incx, y_z, incy);

  status = cublasCcopy(handle, n, x_c, incx, y_c, incy);
  cublasCcopy(handle, n, x_c, incx, y_c, incy);

  status = cublasZcopy(handle, n, x_z, incx, y_z, incy);
  cublasZcopy(handle, n, x_z, incx, y_z, incy);

  cuComplex* result_c = 0;
  cuDoubleComplex* result_z = 0;

  status = cublasCdotu(handle, n, x_c, incx, y_c, incy, result_c);
  cublasCdotu(handle, n, x_c, incx, y_c, incy, result_c);

  status = cublasCdotc(handle, n, x_c, incx, y_c, incy, result_c);
  cublasCdotc(handle, n, x_c, incx, y_c, incy, result_c);

  status = cublasZdotu(handle, n, x_z, incx, y_z, incy, result_z);
  cublasZdotu(handle, n, x_z, incx, y_z, incy, result_z);

  status = cublasZdotc(handle, n, x_z, incx, y_z, incy, result_z);
  cublasZdotc(handle, n, x_z, incx, y_z, incy, result_z);

  status = cublasScnrm2(handle, n, x_c, incx, result_f);
  cublasScnrm2(handle, n, x_c, incx, result_f);

  status = cublasDznrm2(handle, n, x_z, incx, result_d);
  cublasDznrm2(handle, n, x_z, incx, result_d);

  float* c_f = 0;
  float* s_f = 0;
  double* c_d = 0;
  double* s_d = 0;
  cuComplex* c_c = 0;
  cuComplex* s_c = 0;
  cuDoubleComplex* c_z = 0;
  cuDoubleComplex* s_z = 0;

  status = cublasCsrot(handle, n, x_c, incx, y_c, incy, c_f, s_f);
  cublasCsrot(handle, n, x_c, incx, y_c, incy, c_f, s_f);

  status = cublasZdrot(handle, n, x_z, incx, y_z, incy, c_d, s_d);
  cublasZdrot(handle, n, x_z, incx, y_z, incy, c_d, s_d);

  status = cublasCrotg(handle, x_c, y_c, c_f, s_c);
  cublasCrotg(handle, x_c, y_c, c_f, s_c);

  status = cublasZrotg(handle, x_z, y_z, c_d, s_z);
  cublasZrotg(handle, x_z, y_z, c_d, s_z);

  status = cublasCscal(handle, n, alpha_c, x_c, incx);
  cublasCscal(handle, n, alpha_c, x_c, incx);

  status = cublasZscal(handle, n, alpha_z, x_z, incx);
  cublasZscal(handle, n, alpha_z, x_z, incx);

  status = cublasCsscal(handle, n, alpha_f, x_c, incx);
  cublasCsscal(handle, n, alpha_f, x_c, incx);

  status = cublasZdscal(handle, n, alpha_d, x_z, incx);
  cublasZdscal(handle, n, alpha_d, x_z, incx);

  status = cublasCswap(handle, n, x_c, incx, y_c, incy);
  cublasCswap(handle, n, x_c, incx, y_c, incy);

  status = cublasZswap(handle, n, x_z, incx, y_z, incy);
  cublasZswap(handle, n, x_z, incx, y_z, incy);

  //level 2
  int m=0;
  int kl=0;
  int ku=0;
  int lda = 10;
  int trans0 = 0;
  int trans1 = 1;
  int trans2 = 2;
  status = cublasCgbmv(handle, (cublasOperation_t)trans0, m, n, kl, ku, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasCgbmv(handle, CUBLAS_OP_N, m, n, kl, ku, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  status = cublasZgbmv(handle, (cublasOperation_t)trans1, m, n, kl, ku, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZgbmv(handle, CUBLAS_OP_N, m, n, kl, ku, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  status = cublasCgemv(handle, (cublasOperation_t)trans2, m, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasCgemv(handle, CUBLAS_OP_N, m, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  status = cublasZgemv(handle, (cublasOperation_t)0, m, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZgemv(handle, CUBLAS_OP_N, m, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  status = cublasCgeru(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCgeru(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  status = cublasCgerc(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCgerc(handle, m, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  status = cublasZgeru(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZgeru(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);

  status = cublasZgerc(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZgerc(handle, m, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);

  int k = 1;
  int fill0 = 0;
  int fill1 = 1;
  int diag0 = 0;
  int diag1 = 1;
  status = cublasCtbmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)1, (cublasDiagType_t)diag0, n, k, x_c, lda, result_c, incx);
  cublasCtbmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, x_c, lda, result_c, incx);

  status = cublasZtbmv(handle, (cublasFillMode_t)fill1, (cublasOperation_t)2, (cublasDiagType_t)diag1, n, k, x_z, lda, result_z, incx);
  cublasZtbmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, x_z, lda, result_z, incx);

  status = cublasCtbsv(handle, (cublasFillMode_t)0, (cublasOperation_t)trans0, (cublasDiagType_t)0,  n, k, x_c, lda, result_c, incx);
  cublasCtbsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,  n, k, x_c, lda, result_c, incx);

  status = cublasZtbsv(handle, (cublasFillMode_t)1, (cublasOperation_t)trans0, (cublasDiagType_t)1,  n, k, x_z, lda, result_z, incx);
  cublasZtbsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,  n, k, x_z, lda, result_z, incx);

  status = cublasCtpmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, result_c, incx);
  cublasCtpmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, result_c, incx);

  status = cublasZtpmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, result_z, incx);
  cublasZtpmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, result_z, incx);

  status = cublasCtpsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, result_c, incx);
  cublasCtpsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, result_c, incx);

  status = cublasZtpsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, result_z, incx);
  cublasZtpsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, result_z, incx);

  status = cublasCtrmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, lda, result_c, incx);
  cublasCtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, lda, result_c, incx);

  status = cublasZtrmv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, lda, result_z, incx);
  cublasZtrmv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, lda, result_z, incx);

  status = cublasCtrsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_c, lda, result_c, incx);
  cublasCtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_c, lda, result_c, incx);

  status = cublasZtrsv(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, n, x_z, lda, result_z, incx);
  cublasZtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, x_z, lda, result_z, incx);

  status = cublasChemv(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasChemv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  status = cublasZhemv(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZhemv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  status = cublasChbmv(handle, (cublasFillMode_t)fill0, n, k, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);
  cublasChbmv(handle, CUBLAS_FILL_MODE_LOWER, n, k, alpha_c, x_c, lda, x_c, incx, beta_c, y_c, incy);

  status = cublasZhbmv(handle, (cublasFillMode_t)fill0, n, k, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);
  cublasZhbmv(handle, CUBLAS_FILL_MODE_LOWER, n, k, alpha_z, x_z, lda, x_z, incx, beta_z, y_z, incy);

  status = cublasChpmv(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, x_c, incx, beta_c, y_c, incy);
  cublasChpmv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, x_c, incx, beta_c, y_c, incy);

  status = cublasZhpmv(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, x_z, incx, beta_z, y_z, incy);
  cublasZhpmv(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, x_z, incx, beta_z, y_z, incy);

  status = cublasCher(handle, (cublasFillMode_t)fill0, n, alpha_f, x_c, incx, result_c, lda);
  cublasCher(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_f, x_c, incx, result_c, lda);

  status = cublasZher(handle, (cublasFillMode_t)fill0, n, alpha_d, x_z, incx, result_z, lda);
  cublasZher(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_d, x_z, incx, result_z, lda);

  status = cublasCher2(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);
  cublasCher2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, incx, y_c, incy, result_c, lda);

  status = cublasZher2(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);
  cublasZher2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, incx, y_z, incy, result_z, lda);

  status = cublasChpr(handle, (cublasFillMode_t)fill0, n, alpha_f, x_c, incx, result_c);
  cublasChpr(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_f, x_c, incx, result_c);

  status = cublasZhpr(handle, (cublasFillMode_t)fill0, n, alpha_d, x_z, incx, result_z);
  cublasZhpr(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_d, x_z, incx, result_z);

  status = cublasChpr2(handle, (cublasFillMode_t)fill0, n, alpha_c, x_c, incx, y_c, incy, result_c);
  cublasChpr2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_c, x_c, incx, y_c, incy, result_c);

  status = cublasZhpr2(handle, (cublasFillMode_t)fill0, n, alpha_z, x_z, incx, y_z, incy, result_z);
  cublasZhpr2(handle, CUBLAS_FILL_MODE_LOWER, n, alpha_z, x_z, incx, y_z, incy, result_z);

  int N = 100;
  status = cublasCgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans0, N, N, N, alpha_c, x_c, N, y_c, N, beta_c, result_c, N);
  cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha_c, x_c, N, y_c, N, beta_c, result_c, N);

  status = cublasZgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans0, N, N, N, alpha_z, x_z, N, y_z, N, beta_z, result_z, N);
  cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha_z, x_z, N, y_z, N, beta_z, result_z, N);

  cuComplex* A_c = 0;
  cuDoubleComplex* A_z = 0;
  cuComplex* B_c = 0;
  cuDoubleComplex* B_z = 0;
  cuComplex* C_c = 0;
  cuDoubleComplex* C_z = 0;


  int ldb = 10;
  int ldc = 10;


  const float alpha_s = 1;
  const float beta_s = 1;
  const double beta_d = 0;



  int side0 = 0;
  int side1 = 1;
  status = cublasCsymm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasCsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  status = cublasZsymm(handle, (cublasSideMode_t)side1, (cublasFillMode_t)fill0, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  status = cublasCsyrk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_c, A_c, lda, beta_c, C_c, ldc);
  cublasCsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, beta_c, C_c, ldc);

  status = cublasZsyrk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_z, A_z, lda, beta_z, C_z, ldc);
  cublasZsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, beta_z, C_z, ldc);

  status = cublasCsyr2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasCsyr2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  status = cublasZsyr2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZsyr2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  status = cublasCtrsm(handle, (cublasSideMode_t)0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, m, n, alpha_c, A_c, lda, B_c, ldb);
  cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, alpha_c, A_c, lda, B_c, ldb);

  status = cublasZtrsm(handle, (cublasSideMode_t)1, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, m, n, alpha_z, A_z, lda, B_z, ldb);
  cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, alpha_z, A_z, lda, B_z, ldb);

  status = cublasChemm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);
  cublasChemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc);

  status = cublasZhemm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);
  cublasZhemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, m, n, alpha_z, A_z, lda, B_z, ldb, beta_z, C_z, ldc);

  status = cublasCherk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_s, A_c, lda, &beta_s, C_c, ldc);
  cublasCherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &alpha_s, A_c, lda, &beta_s, C_c, ldc);

  status = cublasZherk(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_d, A_z, lda, &beta_d, C_z, ldc);
  cublasZherk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_d, A_z, lda, &beta_d, C_z, ldc);

  status = cublasCher2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_c, A_c, lda, B_c, ldb, &beta_s, C_c, ldc);
  cublasCher2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_c, A_c, lda, B_c, ldb, &beta_s, C_c, ldc);

  status = cublasZher2k(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, alpha_z, A_z, lda, B_z, ldb, &beta_d, C_z, ldc);
  cublasZher2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha_z, A_z, lda, B_z, ldb, &beta_d, C_z, ldc);
}
