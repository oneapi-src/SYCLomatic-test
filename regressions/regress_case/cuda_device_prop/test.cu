// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include<algorithm>
#include <cuda_runtime.h>

__device__ __host__ void cuda_device_prop_test() {
  cudaDeviceProp deviceProp;
  cudaDeviceProp *pDeviceProp = &deviceProp;
  [=]() {
    int a;
    a = std::min((int)pDeviceProp->totalGlobalMem, 1000);
    a = std::min((int)pDeviceProp->integrated, 1000);
    a = std::min((int)pDeviceProp->sharedMemPerBlock, 1000);
    a = std::min((int)pDeviceProp->major, 1000);
    a = std::min((int)pDeviceProp->clockRate, 1000);
    a = std::min((int)pDeviceProp->multiProcessorCount, 1000);
    a = std::min((int)pDeviceProp->warpSize, 1000);
    a = std::min((int)pDeviceProp->maxThreadsPerBlock, 1000);
    a = std::min((int)pDeviceProp->maxThreadsPerMultiProcessor, 1000);
    a = std::min((int)pDeviceProp->maxGridSize[0], 1000);
    a = std::min((int)pDeviceProp->maxThreadsDim[0], 1000);
    char c = std::min(pDeviceProp->name[0], 'A');
  }();
}

int main()
{
    cuda_device_prop_test();
}