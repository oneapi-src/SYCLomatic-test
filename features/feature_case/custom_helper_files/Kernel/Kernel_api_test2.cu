// ====------ Kernel_api_test2.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Kernel/api_test2_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Kernel/api_test2_out/MainSourceFiles.yaml | wc -l > %T/Kernel/api_test2_out/count.txt
// RUN: FileCheck --input-file %T/Kernel/api_test2_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Kernel/api_test2_out

// CHECK: 16
// TEST_FEATURE: Kernel_get_kernel_function_info

__global__ void foo() {}

int main() {
  cudaFuncAttributes attrs;
  cudaFuncGetAttributes(&attrs, foo);
  return 0;
}
