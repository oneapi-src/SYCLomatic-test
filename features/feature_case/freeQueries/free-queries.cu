// ====------ free-queries.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "cooperative_groups.h"

#define TB(b) cg::thread_block b = cg::this_thread_block();

namespace cg = cooperative_groups;
using namespace cooperative_groups;

__global__ void test1() {
  int a = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.x +
          blockDim.x + threadIdx.x;
  cg::thread_block cta = cg::this_thread_block();
  cg::thread_block b0 = cg::this_thread_block(), b1 = cg::this_thread_block();
  cg::sync(cta);
  TB(b);
  a = __syncthreads_and(a);
  __all(a);
  __shfl(a, a);
}

__global__ void test2() {
  int a = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.x +
          blockDim.x + threadIdx.x;
  cg::thread_block cta = cg::this_thread_block();
  cg::thread_block b0 = cg::this_thread_block(), b1 = cg::this_thread_block();
  cg::sync(cta);
  __all(a);
  __shfl(a, a);
}

int main() {
    test1<<<32, 32>>>();

    test2<<<dim3(2,2,32),dim3(2,2,32)>>>();

}

