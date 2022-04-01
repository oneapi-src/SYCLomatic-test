// ====------ util.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cuda_runtime.h> 
#include "util.h"

__global__ void kernel_util(myint a, myint b) {
  myint c= mymax(a,b);  
  printf("kernel_util,%d\n", c); 
}


void run_util(){
 kernel_util<<<1,1>>>(1,2);
}
