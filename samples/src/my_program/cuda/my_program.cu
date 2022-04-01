// ====------ my_program.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <stdio.h>  
#include <cuda_runtime.h>
__global__ void hello_gpu(int n) {
  printf("hello!\n"); 
}

int main(){
  hello_gpu<<<1, 1>>>(1); 
  return 0;
}
