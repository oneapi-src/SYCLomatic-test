// ====------ main.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <stdio.h>  
#include <cuda_runtime.h>

extern void run_util();

__global__ void kernel_main(int n) {
  printf("kernel_main!\n"); 
}

int main(){

  kernel_main<<<1, 1>>>(1); 
  cudaDeviceSynchronize();
  
  run_util();
  cudaDeviceSynchronize(); 
  
  return 0;
}
