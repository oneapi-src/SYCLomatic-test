// ====------ DplExtrasMemory_api_test10.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --use-custom-helper=api -out-root %T/DplExtrasMemory/api_test10_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasMemory/api_test10_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasMemory/api_test10_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasMemory/api_test10_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasMemory/api_test10_out

// CHECK: 8
// TEST_FEATURE: DplExtrasMemory_get_device_pointer

#include <thrust/device_ptr.h>
int main() {
  float* a;
  thrust::device_pointer_cast<float>(a);
  return 0;
}
