// ====------ driverCtx.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
int main(){
  unsigned int ver;
  CUdevice device;
  CUcontext ctx;
  CUdevice* dev_ptr;

  cuDeviceGet(&device, 0);
  cuCtxCreate(&ctx, 0, device);
  cuCtxGetApiVersion(ctx, &ver);
  cuCtxGetDevice(dev_ptr);
  return 0;
}

