// ===-------------- math-cuda-align.cu---------- *- CUDA -* ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cuda_fp8.h>

class __CUDA_ALIGN__(16) color {
    unsigned int r, g, b;
};

int main(){
  color c;
  return 1;
}