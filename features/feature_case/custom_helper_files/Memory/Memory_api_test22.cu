// ====------ Memory_api_test22.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Memory/api_test22_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test22_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test22_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test22_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test22_out

// CHECK: 47
// TEST_FEATURE: Memory_device_memory_assign

static __constant__ const int *schsfirst;

int main() {
  static int *schsfirstD;
  cudaMalloc(&schsfirstD, 16);
  cudaMemcpyToSymbol(schsfirst, &schsfirstD, 8);
  return 0;
}
