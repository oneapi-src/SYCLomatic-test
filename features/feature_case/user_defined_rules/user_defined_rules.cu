// ====------ user_defined_rules.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include<iostream>
#include<cmath>

#define CALL(x) x
#define CALL2(x) x

#if !defined(_MSC_VER)
#define __my_inline__ __forceinline
#else
#define __my_inline__ __inline__ __attribute__((always_inline))
#endif

#define VECTOR int
__forceinline__ __global__ void foo(){
  VECTOR a;
}

int main(){
  int **ptr;
  cudaMalloc(ptr, 50);
  CALL(0);
  return 0;
}
