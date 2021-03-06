// ====------ Image_api_test35.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: dpct --format-range=none  --use-custom-helper=api -out-root %T/Image/api_test35_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test35_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test35_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test35_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test35_out

// CHECK: 44
// usm version of 21
// TEST_FEATURE: Image_image_matrix_to_pitched_data

int main() {
  cudaArray_t a1;
  cudaArray* a2;
  size_t width, height, woffset, hoffset;
  cudaMemcpy2DArrayToArray(a1, woffset, hoffset, a2, woffset, hoffset, width, height, cudaMemcpyDeviceToHost);
  return 0;
}