// ====------ BlasUtils_api_test25.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_gemm_batch

int main() {
  cublasHandle_t handle;
  void * alpha;
  void * beta;
  const void** a;
  const void** b;
  void** c;

  cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, alpha, a, CUDA_R_16F, 4, b, CUDA_R_16F, 4, beta, c, CUDA_R_16F, 4, 2, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
  return 0;
}
