// ====------ Device_api_test20.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test20_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test20_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test20_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test20_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test20_out

// CHECK: 3
// TEST_FEATURE: Device_device_info

int main() {
  cudaDeviceProp p;
  return 0;
}
