// ====------ grid_sync.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cstring>
#include <iostream>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void kernel_1() {
  cg::grid_group grid = cg::this_grid();
  grid.sync();

  printf("kernel_1 run\n");

}


__global__ void kernel_2() {
  cg::grid_group grid_1 = cg::this_grid();
  grid_1.sync();

  cg::grid_group grid_2 = cg::this_grid();
  grid_2.sync();
  printf("kernel_2 run\n");

}

int main() {
  kernel_1<<<4, 4>>>();
  cudaDeviceSynchronize();

  kernel_2<<<4, 4>>>();
  cudaDeviceSynchronize();

  return 0;
}


