// ====------ DplExtrasMemory_api_test7.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasMemory/api_test7_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasMemory/api_test7_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasMemory/api_test7_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasMemory/api_test7_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasMemory/api_test7_out

// CHECK: 14
// TEST_FEATURE: DplExtrasMemory_get_raw_pointer

#include <thrust/device_ptr.h>
int main() {
  double *a;
  thrust::device_ptr<double> d_ptr(a);
  thrust::raw_pointer_cast(d_ptr);
  return 0;
}
