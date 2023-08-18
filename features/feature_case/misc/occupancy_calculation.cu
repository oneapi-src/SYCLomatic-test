// ===------ occupancy_calculation.cu --------------------- *- CUDA -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

__global__ void k() {}

int main() {
  int num_blocks;
  int block_size = 128;
  size_t dynamic_shared_memory_size = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, k, block_size, dynamic_shared_memory_size);
  CUfunction func;
  cuOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, func, block_size, dynamic_shared_memory_size);

  int min_grid_size;
  int block_size_limit;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, k, dynamic_shared_memory_size, block_size_limit);
  return 0;
}
