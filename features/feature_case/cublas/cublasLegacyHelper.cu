// ====------ cublasLegacyHelper.cu---------- *- CUDA -* ----===////
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

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

#define MACRO_A cublasInit()

#define MACRO_B(status) (status)

#define MACRO_C(pointer) status = cublasFree(d_A)

void foo2(cublasStatus){}

void foo(cublasStatus, cublasStatus, cublasStatus, cublasStatus, cublasStatus, cublasStatus, cublasStatus, cublasStatus, cublasStatus, cublasStatus) {}

void bar(cublasStatus_t, cublasStatus_t, cublasStatus_t, cublasStatus_t, cublasStatus_t, cublasStatus_t, cublasStatus_t, cublasStatus_t, cublasStatus_t, cublasStatus_t) {}

cublasStatus foo(int m, int n) {
  return CUBLAS_STATUS_SUCCESS;
}

int main() {
  foo(CUBLAS_STATUS_SUCCESS, CUBLAS_STATUS_NOT_INITIALIZED, CUBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_INVALID_VALUE, CUBLAS_STATUS_ARCH_MISMATCH, CUBLAS_STATUS_MAPPING_ERROR, CUBLAS_STATUS_EXECUTION_FAILED, CUBLAS_STATUS_INTERNAL_ERROR, CUBLAS_STATUS_NOT_SUPPORTED, CUBLAS_STATUS_LICENSE_ERROR);
  bar(CUBLAS_STATUS_SUCCESS, CUBLAS_STATUS_NOT_INITIALIZED, CUBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_INVALID_VALUE, CUBLAS_STATUS_ARCH_MISMATCH, CUBLAS_STATUS_MAPPING_ERROR, CUBLAS_STATUS_EXECUTION_FAILED, CUBLAS_STATUS_INTERNAL_ERROR, CUBLAS_STATUS_NOT_SUPPORTED, CUBLAS_STATUS_LICENSE_ERROR);

  cublasStatus status;
  status = cublasInit();
  cublasInit();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }

  status = MACRO_A;

  int a = sizeof(cublasStatus);
  a = sizeof(cublasStatus_t);
  a = sizeof(cublasHandle_t);
  a = sizeof(cuComplex);
  a = sizeof(cuDoubleComplex);

  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  cublasSetKernelStream(stream1);
  cublasErrCheck(cublasSetKernelStream(stream1));

  float *d_A = NULL;
  int n = 10;
  int elemSize = 4;

  status = cublasAlloc(n, elemSize, (void **)&d_A);
  cublasAlloc(n, elemSize, (void **)&d_A);

  foo2(cublasAlloc(n, elemSize, (void **)&d_A));

  status = cublasFree(d_A);
  cublasFree(d_A);

  foo2(cublasFree(d_A));

  MACRO_B(cublasFree(d_A));

  MACRO_B(cublasGetError());

  MACRO_C(d_A);

  cublasGetError();
  status = cublasGetError();

  foo2(cublasGetError());

  foo2(cublasShutdown());

  foo2(cublasInit());

  status = cublasShutdown();
  cublasShutdown();
  return 0;
}
