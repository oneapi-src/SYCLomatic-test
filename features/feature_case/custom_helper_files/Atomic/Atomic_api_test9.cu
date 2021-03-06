// ====------ Atomic_api_test9.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Atomic/api_test9_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Atomic/api_test9_out/MainSourceFiles.yaml | wc -l > %T/Atomic/api_test9_out/count.txt
// RUN: FileCheck --input-file %T/Atomic/api_test9_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Atomic/api_test9_out

// CHECK: 2
// TEST_FEATURE: Atomic_atomic_fetch_compare_inc

__global__ void test(int *data) {
  int inc = 1;



  atomicInc((unsigned int *)&data[5], (unsigned int)inc);

}
int main() {
  return 0;
}
