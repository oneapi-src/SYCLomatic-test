// ====------ Image_api_test11.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test11_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test11_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test11_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test11_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test11_out

// CHECK: 27
// TEST_FEATURE: Image_image_wrapper_base_get_data

int main() {
  cudaTextureObject_t tex42;
  cudaResourceDesc res42;
  cudaGetTextureObjectResourceDesc(&res42, tex42);
  return 0;
}
