// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda_runtime.h>

int main(){
  int val, dev_id;
  
  cudaGetDevice(&dev_id);
  cudaDeviceAttr attr = cudaDevAttrComputeCapabilityMajor;
  cudaDeviceGetAttribute(&val, attr, dev_id);

  return 0;

}