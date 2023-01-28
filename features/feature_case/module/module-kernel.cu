// ====------ module-kernel.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
static texture<int, 3> tex;

extern "C" __global__ void foo(float* k, float* y);

extern "C" __global__ void foo(float* k, float* y){
    extern __shared__ int s[];
    int a = threadIdx.x;
}

void goo(){
    float *a, *b;
    foo<<<1,2>>>(a, b);
}
