// ====------ cublasReturnType.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

cublasStatus_t foo(int m, int n) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasHandle_t foo1(int m) {
  return 0;
}

cuComplex foo2(cuComplex m) {
  return make_cuComplex(1, 0);
}

cuDoubleComplex foo3(cuDoubleComplex m) {
  return make_cuDoubleComplex(1, 0);
}

cublasOperation_t foo4(cublasOperation_t m) {
  return CUBLAS_OP_C;
}

cublasFillMode_t foo5(cublasFillMode_t m) {
  return CUBLAS_FILL_MODE_LOWER;
}

cublasSideMode_t foo6(cublasSideMode_t m) {
  return CUBLAS_SIDE_RIGHT;
}

cublasDiagType_t foo7(cublasDiagType_t m) {
  return CUBLAS_DIAG_NON_UNIT;
}
