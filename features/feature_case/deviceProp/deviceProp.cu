// ====------ deviceProp.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

int main() {
  cudaDeviceProp deviceProp;
  int a1 = deviceProp.maxThreadsDim[2];
  int a2 = deviceProp.memPitch;
  int a3 = deviceProp.totalConstMem;
  // This properity is not supported.
  // int a4 = deviceProp.regsPerBlock;
  int a5 = deviceProp.textureAlignment;
  int a6 = deviceProp.kernelExecTimeoutEnabled;
  int a7 = deviceProp.ECCEnabled;
  int freq = deviceProp.memoryClockRate;
  int buswidth = deviceProp.memoryBusWidth;
  size_t share_multi_proc_mem_size = deviceProp.sharedMemPerMultiprocessor;
  return 0;
}
