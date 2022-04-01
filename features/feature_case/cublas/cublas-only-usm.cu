// ====------ cublas-only-usm.cu---------- *- CUDA -* ----===////
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

int main(){
  cublasHandle_t handle;
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
  cublasOperation_t trans3 = CUBLAS_OP_N;
  int N = 10;
  float alpha_S;
  float beta_S;
  double alpha_D;
  double beta_D;
  cuComplex alpha_C;
  cuComplex beta_C;
  cuDoubleComplex alpha_Z;
  cuDoubleComplex beta_Z;

  int a = cublasSgemmBatched(handle, trans3, trans3, N, N, N, &alpha_S, d_A_S_array, N, d_B_S_array, N, &beta_S, d_C_S_array, N, 10);
  cublasDgemmBatched(handle, trans3, trans3, N, N, N, &alpha_D, d_A_D_array, N, d_B_D_array, N, &beta_D, d_C_D_array, N, 10);
  cublasCgemmBatched(handle, trans3, trans3, N, N, N, &alpha_C, d_A_C_array, N, d_B_C_array, N, &beta_C, d_C_C_array, N, 10);
  cublasZgemmBatched(handle, trans3, trans3, N, N, N, &alpha_Z, d_A_Z_array, N, d_B_Z_array, N, &beta_Z, d_C_Z_array, N, 10);

  a = cublasStrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, trans3, CUBLAS_DIAG_UNIT, N, N, &alpha_S, d_A_S_array, N, d_C_S_array, N, 10);
  cublasDtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, trans3, CUBLAS_DIAG_UNIT, N, N, &alpha_D, d_A_D_array, N, d_C_D_array, N, 10);
  cublasCtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, trans3, CUBLAS_DIAG_UNIT, N, N, &alpha_C, d_A_C_array, N, d_C_C_array, N, 10);
  cublasZtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, trans3, CUBLAS_DIAG_UNIT, N, N, &alpha_Z, d_A_Z_array, N, d_C_Z_array, N, 10);
}
