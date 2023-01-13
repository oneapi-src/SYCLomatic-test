// ====------ device_info.cu---------- *- CUDA -* -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===--------------------------------------------------------------===//

#include <stdio.h>

int main() {
  // no need to use `cudaSetDevice`
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);

  printf("total_mem : [%lu]\n", total_mem);
  printf("free_mem  : [%lu]\n", free_mem);
  return 0;
}