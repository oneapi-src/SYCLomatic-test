// ====------ BlasUtils_api_test9.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: dpct --format-range=none  --use-custom-helper=api -out-root %T/BlasUtils/api_test9_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/BlasUtils/api_test9_out/MainSourceFiles.yaml | wc -l > %T/BlasUtils/api_test9_out/count.txt
// RUN: FileCheck --input-file %T/BlasUtils/api_test9_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/BlasUtils/api_test9_out

// CHECK: 26

#include "cublas_v2.h"

// TEST_FEATURE: BlasUtils_getri_batch_wrapper

int main() {
  cublasHandle_t handle;
  int n = 275;
  int lda = 275;
  int ldc = 275;

  float **Carray_S = 0;
  int *PivotArray = 0;
  int *infoArray = 0;
  int batchSize = 10;
  const float **Aarray_Sc = 0;

  cublasSgetriBatched(handle, n, Aarray_Sc, lda, PivotArray, Carray_S, ldc, infoArray, batchSize);

  return 0;
}
