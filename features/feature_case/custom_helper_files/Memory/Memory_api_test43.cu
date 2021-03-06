// ====------ Memory_api_test43.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: dpct --format-range=none   --use-custom-helper=api -out-root %T/BlasUtils/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/BlasUtils/api_test1_out/MainSourceFiles.yaml | wc -l > %T/BlasUtils/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/BlasUtils/api_test1_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/BlasUtils/api_test1_out

// CHECK: 20

// TEST_FEATURE: Memory_async_dpct_free

#include "cublas_v2.h"
#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t cusolverH;
  float B_f, C_f = 0;
  int devInfo;
  cusolverDnSpotrs(cusolverH, CUBLAS_FILL_MODE_LOWER, 0, 0, &C_f, 4, &B_f, 4, &devInfo);
  return 0;
}
