// ====------ DplExtrasAlgorithm_api_test6.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test6_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test6_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test6_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test6_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test6_out

// CHECK: 5
// TEST_FEATURE: DplExtrasAlgorithm_stable_sort

#include <thrust/sort.h>
int main() {
  int *a;
  thrust::stable_sort_by_key(a, a + 10, a);
  return 0;
}
