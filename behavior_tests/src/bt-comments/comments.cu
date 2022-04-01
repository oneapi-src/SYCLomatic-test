// ====------ comments.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
static texture<uint2, 1> tex21;

__constant__ int a = 1;
__device__ int b[36][36];

__device__ void test() {
  __shared__ int cl[36];
  cl[0] = b[0][0] + a;
}

__global__ void kernel() {
  test();
  __device__ uint2 al[16];
  __shared__ int bl[12][12];
  al[0] = tex1D(tex21, 1);
  bl[0][0] = 0;
  printf("test\n");
}

int main() {
    cudaMemcpy3DParms p;
    dim3 griddim(1, 2, 3);
    dim3 threaddim(1, 2, 3);
    kernel<<<griddim, threaddim>>>();
}

