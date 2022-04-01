// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <stdio.h>

#include <stdlib.h>

#include <math.h>

#include <assert.h>

 

#include <cuda_runtime.h>

 

texture<uint8_t, 1, cudaReadModeNormalizedFloat> texref;

int main()
{        
  uint8_t* dev_in;
  int size = 1920*1080*6;
  cudaBindTexture(0, texref, dev_in, sizeof(uint8_t)*size);
  cudaUnbindTexture(&texref); 
  return 0;
}
