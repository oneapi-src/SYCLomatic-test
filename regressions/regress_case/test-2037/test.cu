// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda.h>
#include <cuda_runtime_api.h>
#define CUDA_SAFE_CALL( call) do {\
  int err = call;                \
} while (0)
int main(){

  CUdevice device;

  cuDeviceGet(&device, 0);

  char name[100];

  cuDeviceGetName(name, 90, device);

  CUDA_SAFE_CALL(cuDeviceGetName(name, 90, device));

  return 0;
}