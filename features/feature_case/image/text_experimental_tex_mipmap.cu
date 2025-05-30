// ===------------- text_experimental_tex_mipmap.cu ------ *- CUDA -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//
#include <cuda.h>
#include <iostream>

const int height = 2;
const int width  = 4;
const int depth  = 2;

void set_3D_descriptor(CUDA_ARRAY3D_DESCRIPTOR &desc) {
  desc.Width = width;
  desc.Depth = depth;
  desc.Height = height;
  desc.Format = CU_AD_FORMAT_SIGNED_INT16;
  desc.NumChannels = 4;
}

int main() {
  CUdevice device;
  CUcontext context;

  // Initialize CUDA
  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    std::cerr << "Failed to initialize CUDA\n";
    return -1;
  }

  // Get the device
  result = cuDeviceGet(&device, 0);
  if (result != CUDA_SUCCESS) {
    std::cerr << "Failed to get CUDA device\n";
    return -1;
  }

  // Create a context
  result = cuCtxCreate(&context, 0, device);
  if (result != CUDA_SUCCESS) {
    std::cerr << "Failed to create CUDA context\n";
    return -1;
  }

  CUDA_ARRAY3D_DESCRIPTOR desc;
  set_3D_descriptor(desc);

  CUmipmappedArray mmArray;
  unsigned int numMipmapLevels = 2;
  result = cuMipmappedArrayCreate(&mmArray, &desc, numMipmapLevels);
  if (result != CUDA_SUCCESS) {
    std::cerr << "Failed to create mipmapped array\n";
    return -1;
  }

  CUarray level_arr;
  result = cuMipmappedArrayGetLevel(&level_arr, mmArray, 0);  // Get level 0
  if (result != CUDA_SUCCESS) {
    std::cerr << "Failed to get mipmap level\n";
    return -1;
  }
  
  short4 mm1[height * width * depth] = {
    {1,  2, 3, 4},   {5, 6, 7, 8},   {9, 10, 11, 12},   {13, 14, 15, 16},
    {17, 18, 19, 20},  {21, 22, 23, 24}, {25, 26, 27, 28}, {29, 30, 31, 32},

    {33, 34, 35, 36}, {37, 38, 39, 40}, {41, 42, 43, 44}, {45, 46, 47, 48},
    {49, 50, 51, 52}, {53, 54, 55, 56}, {57, 58, 59, 60}, {61, 62, 63, 64}
  };

  CUDA_MEMCPY3D copyAssist{0};
  // specify source details
  copyAssist.srcHost = mm1;
  copyAssist.srcMemoryType = CU_MEMORYTYPE_HOST;
  copyAssist.srcHeight = height;
  copyAssist.srcPitch = sizeof(short4) * width;

  // specify destination details
  copyAssist.dstArray = level_arr;
  copyAssist.dstMemoryType = CU_MEMORYTYPE_ARRAY;

  // specify copy dimensions
  copyAssist.WidthInBytes = sizeof(short4) * width;
  copyAssist.Height = height;
  copyAssist.Depth = depth;

  result = cuMemcpy3D(&copyAssist);
  if (result != CUDA_SUCCESS) {
    std::cerr << "Copy from host to device failed for mipmaped array\n";
    return -1;
  }

  CUtexref texRef{0};
  cuTexRefCreate(&texRef);
  cuTexRefSetMipmappedArray(texRef, mmArray, 0);

  cuTexRefSetMipmapFilterMode(texRef, CU_TR_FILTER_MODE_POINT);

  CUfilter_mode fm;
  cuTexRefGetMipmapFilterMode(&fm, texRef);
  if (fm != CU_TR_FILTER_MODE_POINT) {
    std::cout << "Filter mode test failed";
    return -1;
  }

  float min_clamp, max_clamp;
  cuTexRefGetMipmapLevelClamp(&min_clamp, &max_clamp, texRef);

  CUmipmappedArray anotherArray;
  cuTexRefGetMipmappedArray(&anotherArray, texRef);
  cuMipmappedArrayDestroy(mmArray);

  return 0;
}
