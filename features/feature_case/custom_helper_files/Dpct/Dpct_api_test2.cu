// ====------ Dpct_api_test2.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: dpct --format-range=none --sycl-named-lambda  --use-custom-helper=api -out-root %T/Dpct/api_test2_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Dpct/api_test2_out/MainSourceFiles.yaml | wc -l > %T/Dpct/api_test2_out/count.txt
// RUN: cat %T/Dpct/api_test2_out/include/dpct/dpct.hpp >> %T/Dpct/api_test2_out/count.txt
// RUN: FileCheck --input-file %T/Dpct/api_test2_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Dpct/api_test2_out

// CHECK: 17

// CHECK: #ifdef DPCT_NAMED_LAMBDA
// CHECK-NEXT: template <class... Args> class dpct_kernel_name;
// CHECK-NEXT: template <int Arg> class dpct_kernel_scalar;
// CHECK-NEXT: #endif

// TEST_FEATURE: Dpct_dpct_named_lambda

__global__ void foo() {}

int main() {
  foo<<<1, 1>>>();
  return 0;
}
