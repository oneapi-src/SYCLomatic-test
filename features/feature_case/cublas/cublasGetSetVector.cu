// ====------ cublasGetSetVector.cu---------- *- CUDA -* ----===////
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

constexpr int foo(int i) {
  return i;
}

int main() {
  cublasStatus_t status;
  cublasHandle_t handle;
  status = cublasCreate(&handle);
  int N = 275;
  float *h_C;
  float *d_C;
  float *h_A;
  float *d_A;
  cudaStream_t stream;

  status = cublasGetVector(N, sizeof(h_C[0]), d_C, 1, h_C, 1);

#define INCX_MARCO 1
  const int ConstIncy = 1;
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, INCX_MARCO, d_A, ConstIncy);

  cublasGetVector(N, sizeof(h_C[0]), d_C, 1, h_C, 1);

  cublasSetVector(N, sizeof(h_A[0]), h_A, 1, d_A, 1);

  cublasGetVector(N, sizeof(h_C[0]), d_C, 2, h_C, 1);

#define INCY_MARCO 2
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, 1, d_A, INCY_MARCO);

  const int ConstIncx = 2;
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, ConstIncx, d_A, INCY_MARCO);

  int incx = 1;
  int incy = 1;

  status = cublasSetVector(N, sizeof(h_A[0]), h_A, incx, d_A, incy);

  const int ConstIncxNE = incx;
  const int ConstIncyNE = incy;
  status = cublasSetVector(N, sizeof(h_A[0]), h_A, ConstIncxNE, d_A, ConstIncyNE);

  const int ConstIncxT = 1;
  const int ConstIncyT = 1;
  constexpr int ConstExprIncx = 3;
  constexpr int ConstExprIncy = 3;
  cublasSetVector(N, sizeof(h_A[0]), h_A, foo(incx), d_A, foo(incy));

  cublasSetVector(N, sizeof(h_A[0]), h_A, foo(ConstIncxT), d_A, foo(ConstIncyT));

  cublasGetVector(N, sizeof(h_A[0]), h_A, foo(ConstExprIncx), d_A, ConstExprIncy);

  status = cublasGetVectorAsync(N, sizeof(h_C[0]), d_C, 1, h_C, 1, stream);
  cublasGetVectorAsync(N, sizeof(h_C[0]), d_C, 1, h_C, 1, stream);

  status = cublasSetVectorAsync(N, sizeof(h_C[0]), d_C, 1, h_C, 1, stream);
  cublasSetVectorAsync(N, sizeof(h_C[0]), d_C, 1, h_C, 1, stream);

  return 0;
}
