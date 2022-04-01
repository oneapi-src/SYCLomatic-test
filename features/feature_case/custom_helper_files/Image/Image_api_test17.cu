// ====------ Image_api_test17.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test17_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test17_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test17_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test17_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test17_out

// CHECK: 28
// TEST_FEATURE: Image_image_wrapper_base_set_filtering_mode

int main() {
  CUtexref tex;
  CUfilter_mode filter_mode;
  cuTexRefSetFilterMode(tex, filter_mode);
  return 0;
}
