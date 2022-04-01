// ====------ Device_api_test23.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test23_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test23_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test23_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test23_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test23_out

// CHECK: 32
// TEST_FEATURE: Device_device_ext_get_max_clock_frequency

int main() {
  int res;
  cudaDeviceGetAttribute(&res, cudaDevAttrClockRate, 0);
  return 0;
}
