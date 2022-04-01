// ====------ cublas-create-Sgemm-destroy.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cstdio>
#include "cublas_v2.h"
#include <cuda_runtime.h>

void foo (cublasStatus_t s){
}
cublasStatus_t bar (cublasStatus_t s){
  return s;
}

extern cublasHandle_t handle2;

int foo2(cudaDataType DT) {
  cublasStatus_t status;
  cublasHandle_t handle;
  cublasCreate(&handle);
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }

  cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
  cublasGetPointerMode(handle, &mode);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  cudaDataType_t cdt;
  cublasDataType_t cbdt;

  cublasAtomicsMode_t Atomicsmode;
  cublasGetAtomicsMode(handle, &Atomicsmode);
  cublasSetAtomicsMode(handle, Atomicsmode);

  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  cublasSetStream(handle, stream1);
  status = cublasSetStream(handle, stream1);
  cublasGetStream(handle, &stream1);
  status = cublasGetStream(handle, &stream1);


  int N = 275;
  float *d_A_S = 0;
  float *d_B_S = 0;
  float *d_C_S = 0;
  float alpha_S = 1.0f;
  float beta_S = 0.0f;
  int trans0 = 0;
  int trans1 = 1;
  int trans2 = 2;
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  double *d_A_D = 0;
  double *d_B_D = 0;
  double *d_C_D = 0;
  double alpha_D = 1.0;
  double beta_D = 0.0;
  status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);
  cublasDgemm(handle, (cublasOperation_t)trans2, (cublasOperation_t)2, N, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);

  __half *d_A_H = 0;
  __half *d_B_H = 0;
  __half *d_C_H = 0;
  __half alpha_H;
  __half beta_H;
  status = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N);
  cublasHgemm(handle, (cublasOperation_t)trans2, (cublasOperation_t)2, N, N, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N);

  void *alpha, *beta, *A, *B, *C;

  status = cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha, A, CUDA_R_16F, N, B, CUDA_R_16F, N, beta, C, CUDA_R_16F, N, CUDA_R_16F, CUBLAS_GEMM_ALGO0);
  cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha, A, CUDA_R_16F, N, B, CUDA_R_16F, N, beta, C, CUDA_R_16F, N, CUDA_R_32F, CUBLAS_GEMM_ALGO0);
  cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha, A, CUDA_R_16F, N, B, CUDA_R_16F, N, beta, C, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_ALGO0);
  cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha, A, CUDA_R_32F, N, B, CUDA_R_32F, N, beta, C, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_ALGO0);
  cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha, A, CUDA_R_64F, N, B, CUDA_R_64F, N, beta, C, CUDA_R_64F, N, CUDA_R_64F, CUBLAS_GEMM_ALGO0);
  cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha, A, CUDA_C_32F, N, B, CUDA_C_32F, N, beta, C, CUDA_C_32F, N, CUDA_C_32F, CUBLAS_GEMM_ALGO0);
  cublasGemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, alpha, A, CUDA_C_64F, N, B, CUDA_C_64F, N, beta, C, CUDA_C_64F, N, CUDA_C_64F, CUBLAS_GEMM_ALGO0);

  float2 alpha_C, beta_C;
  cublasSgemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, &alpha_S, A, CUDA_R_16F, N, B, CUDA_R_16F, N, &beta_S, C, CUDA_R_16F, N);
  cublasSgemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, &alpha_S, A, CUDA_R_16F, N, B, CUDA_R_16F, N, &beta_S, C, CUDA_R_32F, N);
  cublasSgemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, &alpha_S, A, CUDA_R_32F, N, B, CUDA_R_32F, N, &beta_S, C, CUDA_R_32F, N);
  cublasCgemmEx(handle, CUBLAS_OP_C, CUBLAS_OP_C, N, N, N, &alpha_C, A, CUDA_C_32F, N, B, CUDA_C_32F, N, &beta_C, C, CUDA_C_32F, N);

  for (;;) {
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
    beta_S = beta_S + 1;
  }
  alpha_S = alpha_S + 1;

  for (;;) {
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
    beta_S = beta_S + 1;
  }
  alpha_S = alpha_S + 1;


  foo(bar(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)));

  status = cublasDestroy(handle);
  cublasDestroy(handle);
  return 0;
}

void foo4() {
  cublasHandle_t handle;
  float   *a_f, *b_f, *x_f, *c_f, *alpha_f, *beta_f;
  double  *a_d, *b_d, *x_d, *c_d, *alpha_d, *beta_d;
  float2  *a_c, *b_c, *x_c, *c_c, *alpha_c, *beta_c;
  double2 *a_z, *b_z, *x_z, *c_z, *alpha_z, *beta_z;

  cublasSdgmm(handle, CUBLAS_SIDE_LEFT, 2, 2, a_f, 2, x_f, 1, c_f, 2);
  cublasDdgmm(handle, CUBLAS_SIDE_LEFT, 2, 2, a_d, 2, x_d, 1, c_d, 2);
  cublasCdgmm(handle, CUBLAS_SIDE_LEFT, 2, 2, a_c, 2, x_c, 1, c_c, 2);
  cublasZdgmm(handle, CUBLAS_SIDE_LEFT, 2, 2, a_z, 2, x_z, 1, c_z, 2);
}
