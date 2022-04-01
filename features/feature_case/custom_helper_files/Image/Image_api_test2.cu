// ====------ Image_api_test2.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test2_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test2_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test2_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test2_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test2_out

// CHECK: 14
// TEST_FEATURE: Image_image_matrix

int main() {
  cudaArray* a;
  return 0;
}
