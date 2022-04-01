// ====------ cublas-usm.cu---------- *- CUDA -* ----===////
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

cublasHandle_t handle;
int N = 275;
float *h_a, *h_b, *h_c;
const float *d_A_S;
const float *d_B_S;
float *d_C_S;
float alpha_S = 1.0f;
float beta_S = 0.0f;
int trans0 = 0;
int trans1 = 1;
int trans2 = 2;
int fill0 = 0;
int side0 = 0;
int diag0 = 0;
int *result = 0;
const float *x_S = 0;
const float *y_S = 0;

const double *d_A_D;
const double  *d_B_D;
double  *d_C_D;
double alpha_D;
double beta_D;
const double *x_D;
const double *y_D;

const float2 *d_A_C;
const float2  *d_B_C;
float2  *d_C_C;
float2 alpha_C;
float2 beta_C;
const float2 *x_C;
const float2 *y_C;

const double2 *d_A_Z;
const double2  *d_B_Z;
double2  *d_C_Z;
double2 alpha_Z;
double2 beta_Z;
const double2 *x_Z;
const double2 *y_Z;

float* result_S;
double* result_D;
float2* result_C;
double2* result_Z;

int incx, incy, lda, ldb, ldc;

int main() {

  int a = cublasSetVector(10, sizeof(float), h_a, 11111, d_C_S, 11111);
  cublasSetVector(10, sizeof(float), h_b, 1, d_C_S, 1);
  cublasSetVector(10, sizeof(float), h_c, 1, d_C_S, 1);
  a = cublasSetMatrix(100, 100, 10000, h_a, 100, d_C_S, 100);


  cublasPointerMode_t mode = CUBLAS_POINTER_MODE_DEVICE;
  cublasGetPointerMode(handle, &mode);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

  //level 1

  a = cublasIsamax(handle, N, x_S, N, result);
  cublasIdamax(handle, N, x_D, N, result);
  a = cublasIcamax(handle, N, x_C, N, result);
  cublasIzamax(handle, N, x_Z, N, result);

  a = cublasIsamin(handle, N, x_S, N, result);
  cublasIdamin(handle, N, x_D, N, result);
  a = cublasIcamin(handle, N, x_C, N, result);
  cublasIzamin(handle, N, x_Z, N, result);

  a = cublasSrotm(handle, N, d_C_S, N, d_C_S, N, x_S);
  cublasDrotm(handle, N, d_C_D, N, d_C_D, N, x_D);

  a = cublasScopy(handle, N, x_S, incx, d_C_S, incy);
  cublasDcopy(handle, N, x_D, incx, d_C_D, incy);
  a = cublasCcopy(handle, N, x_C, incx, d_C_C, incy);
  cublasZcopy(handle, N, x_Z, incx, d_C_Z, incy);


  a = cublasSaxpy(handle, N, &alpha_S, x_S, incx, result_S, incy);
  cublasDaxpy(handle, N, &alpha_D, x_D, incx, result_D, incy);
  a = cublasCaxpy(handle, N, &alpha_C, x_C, incx, result_C, incy);
  cublasZaxpy(handle, N, &alpha_Z, x_Z, incx, result_Z, incy);

  a = cublasSscal(handle, N, &alpha_S, result_S, incx);
  cublasDscal(handle, N, &alpha_D, result_D, incx);
  a = cublasCscal(handle, N, &alpha_C, result_C, incx);
  cublasZscal(handle, N, &alpha_Z, result_Z, incx);

  a = cublasSnrm2(handle, N, x_S, incx, result_S);
  cublasDnrm2(handle, N, x_D, incx, result_D);
  a = cublasScnrm2(handle, N, x_C, incx, result_S);
  cublasDznrm2(handle, N, x_Z, incx, result_D);

  a = cublasSasum(handle, N, x_S, incx, result_S);
  cublasDasum(handle, N, x_D, incx, result_D);
  a = cublasScasum(handle, N, x_C, incx, result_S);
  cublasDzasum(handle, N, x_Z, incx, result_D);

  float *a_S, *b_S, *c_S, *s_S;
  double *a_D, *b_D, *c_D, *s_D;
  float2 *a_C, *b_C, *s_C;
  double2 *a_Z, *b_Z, *s_Z;

  a = cublasSrotg(handle, a_S, b_S, c_S, s_S);
  cublasDrotg(handle, a_D, b_D, c_D, s_D);
  a = cublasCrotg(handle, a_C, b_C, c_S, s_C);
  cublasZrotg(handle, a_Z, b_Z, c_D, s_Z);

  const float *y1_S;
  const double *y1_D;
  a = cublasSrotmg(handle, a_S, b_S, c_S, y1_S, s_S);
  cublasDrotmg(handle, a_D, b_D, c_D, y1_D, s_D);


  a = cublasSdot(handle, N, x_S, incx, y_S, incy, result_S);
  cublasDdot(handle, N, x_D, incx, y_D, incy, result_D);

  a = cublasCdotc(handle, N, x_C, incx, y_C, incy, result_C);
  cublasZdotc(handle, N, x_Z, incx, y_Z, incy, result_Z);

  a = cublasCdotu(handle, N, x_C, incx, y_C, incy, result_C);
  cublasZdotu(handle, N, x_Z, incx, y_Z, incy, result_Z);

  //level 2

  a = cublasSgemv(handle, (cublasOperation_t)trans2, N, N, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  cublasDgemv(handle, CUBLAS_OP_N, N, N, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  a = cublasCgemv(handle, (cublasOperation_t)trans2, N, N, &alpha_C, x_C, lda, y_C, incx, &beta_C, result_C, incy);
  cublasZgemv(handle, CUBLAS_OP_N, N, N, &alpha_Z, x_Z, lda, y_Z, incx, &beta_Z, result_Z, incy);

  a = cublasSger(handle, N, N, &alpha_S, x_S, incx, y_S, incy, result_S, lda);
  cublasDger(handle, N, N, &alpha_D, x_D, incx, y_D, incy, result_D, lda);
  a = cublasCgeru(handle, N, N, &alpha_C, x_C, incx, y_C, incy, result_C, lda);
  cublasCgerc(handle, N, N, &alpha_C, x_C, incx, y_C, incy, result_C, lda);
  a = cublasZgeru(handle, N, N, &alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);
  cublasZgerc(handle, N, N, &alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);








  //level 3

  a = cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  cublasDgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);
  a = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_C, d_A_C, N, d_B_C, N, &beta_C, d_C_C, N);
  cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_Z, d_A_Z, N, d_B_Z, N, &beta_Z, d_C_Z, N);

  a = cublasSgemmStridedBatched(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, 16, d_B_S, N, 16, &beta_S, d_C_S, N, 16, 10);
  cublasDgemmStridedBatched(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_D, d_A_D, N, 16, d_B_D, N, 16, &beta_D, d_C_D, N, 16, 10);
  a = cublasCgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_C, d_A_C, N, 16, d_B_C, N, 16, &beta_C, d_C_C, N, 16, 10);
  cublasZgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_Z, d_A_Z, N, 16, d_B_Z, N, 16, &beta_Z, d_C_Z, N, 16, 10);

  __half *d_A_H = 0;
  __half *d_B_H = 0;
  __half *d_C_H = 0;
  __half alpha_H;
  __half beta_H;
  cublasOperation_t trans3 = CUBLAS_OP_N;
  a = cublasHgemm(handle, trans3, trans3, N, N, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N);

  void *alpha, *beta, *A, *B, *C;
  cublasGemmAlgo_t algo = CUBLAS_GEMM_ALGO0;
  
  cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha, A, CUDA_R_32F, N, B, CUDA_R_32F, N, beta, C, CUDA_R_32F, N, CUDA_R_32F, algo);

  float2 alpha_C, beta_C;
  cublasSgemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, &alpha_S, A, CUDA_R_16F, N, B, CUDA_R_16F, N, &beta_S, C, CUDA_R_16F, N);
  cublasSgemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, &alpha_S, A, CUDA_R_16F, N, B, CUDA_R_16F, N, &beta_S, C, CUDA_R_32F, N);
  cublasSgemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, &alpha_S, A, CUDA_R_32F, N, B, CUDA_R_32F, N, &beta_S, C, CUDA_R_32F, N);
  cublasCgemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, &alpha_C, A, CUDA_C_32F, N, B, CUDA_C_32F, N, &beta_C, C, CUDA_C_32F, N);

  const float** d_A_S_array;
  const float** d_B_S_array;
  float** d_C_S_array;
  const double** d_A_D_array;
  const double** d_B_D_array;
  double** d_C_D_array;
  const cuComplex** d_A_C_array = 0;
  const cuComplex** d_B_C_array = 0;
  cuComplex** d_C_C_array = 0;
  const cuDoubleComplex** d_A_Z_array = 0;
  const cuDoubleComplex** d_B_Z_array = 0;
  cuDoubleComplex** d_C_Z_array = 0;

  a = cublasStrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_S, d_A_S, N, d_B_S, N, d_C_S, N);
  cublasDtrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_D, d_A_D, N, d_B_D, N, d_C_D, N);
  a = cublasCtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, N, N, &alpha_C, d_A_C, N, d_B_C, N, d_C_C, N);
  cublasZtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, N, N, &alpha_Z, d_A_Z, N, d_B_Z, N, d_C_Z, N);


  a = cublasSsyrkx(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans1, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  cublasDsyrkx(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans1, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);



  if(int stat = cublasStrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_S, d_A_S, N, d_B_S, N, d_C_S, N)){}

  if(int stat = cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)){}


}

int foo1(){
  return cublasStrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_S, d_A_S, N, d_B_S, N, d_C_S, N);
}

int foo2(){
  return cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
}

void foo3() {
  cublasHandle_t handle;
  float   *a_f, *b_f, *x_f, *c_f, *alpha_f, *beta_f;
  double  *a_d, *b_d, *x_d, *c_d, *alpha_d, *beta_d;
  float2  *a_c, *b_c, *x_c, *c_c, *alpha_c, *beta_c;
  double2 *a_z, *b_z, *x_z, *c_z, *alpha_z, *beta_z;

  cublasSsyrkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, 2, 3, alpha_f, a_f, 3, b_f, 3, beta_f, c_f, 2);
  cublasDsyrkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, 2, 3, alpha_d, a_d, 3, b_d, 3, beta_d, c_d, 2);
  cublasCsyrkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, 2, 3, alpha_c, a_c, 3, b_c, 3, beta_c, c_c, 2);
  cublasZsyrkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, 2, 3, alpha_z, a_z, 3, b_z, 3, beta_z, c_z, 2);
  cublasCherkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, 2, 3, alpha_c, a_c, 3, b_c, 3, beta_f, c_c, 2);
  cublasZherkx(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, 2, 3, alpha_z, a_z, 3, b_z, 3, beta_d, c_z, 2);

  cublasSdgmm(handle, CUBLAS_SIDE_LEFT, 2, 2, a_f, 2, x_f, 1, c_f, 2);
  cublasDdgmm(handle, CUBLAS_SIDE_LEFT, 2, 2, a_d, 2, x_d, 1, c_d, 2);
  cublasCdgmm(handle, CUBLAS_SIDE_LEFT, 2, 2, a_c, 2, x_c, 1, c_c, 2);
  cublasZdgmm(handle, CUBLAS_SIDE_LEFT, 2, 2, a_z, 2, x_z, 1, c_z, 2);
}
