// ====------ device_cpu.cu---------------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------===//

void bar(const void *devPtr, int size) {
  cudaMemAdvise(devPtr, size, cudaMemAdviseSetPreferredLocation,
                cudaCpuDeviceId);
}

int main() {
  int *devPtr;
  bar(devPtr, 100);
  return 0;
}
