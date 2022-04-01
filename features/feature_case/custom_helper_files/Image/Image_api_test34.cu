// ====------ Image_api_test34.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test34_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test34_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test34_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test34_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test34_out

// CHECK: 30
// TEST_FEATURE: Image_image_data_get_data_type
// TEST_FEATURE: Image_image_data_get_channel_num
// TEST_FEATURE: Image_image_data_get_channel_type
// TEST_FEATURE: Image_image_data_get_data_ptr
// TEST_FEATURE: Image_image_data_get_pitch
// TEST_FEATURE: Image_image_data_get_x
// TEST_FEATURE: Image_image_data_get_y
// TEST_FEATURE: Image_image_data_get_channel

int main() {
  CUDA_RESOURCE_DESC res42;
  CUresourcetype a = res42.resType;
  int num;
  num = res42.res.pitch2D.numChannels;
  CUarray_format format = res42.res.pitch2D.format;
  CUdeviceptr ptr = res42.res.pitch2D.devPtr;
  num = res42.res.pitch2D.pitchInBytes;
  num = res42.res.pitch2D.width;
  num = res42.res.pitch2D.height;

  cudaResourceDesc res;
  cudaChannelFormatDesc desc = res.res.pitch2D.desc;
  return 0;
}
