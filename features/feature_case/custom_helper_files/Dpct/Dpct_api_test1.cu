// ====------ Dpct_api_test1.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Dpct/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Dpct/api_test1_out/MainSourceFiles.yaml | wc -l > %T/Dpct/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/Dpct/api_test1_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Dpct/api_test1_out

// CHECK: 2

// TEST_FEATURE: Dpct_non_local_include_dependency
// TEST_FEATURE: Dpct_dpct_align_and_inline


class __align__(8) T1 {
    unsigned int l, a;
};

__forceinline__ void foo(){}

int main() {
  return 0;
}
