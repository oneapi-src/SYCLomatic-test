// ====------ hellocuda.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <stdio.h>

__global__ void hellocuda(int n) { printf("hello cuda!\n"); }

int main() {
  hellocuda<<<1, 1>>>(1);
  return 0;
}
