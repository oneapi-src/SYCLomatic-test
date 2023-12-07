// ====------ cooperative_groups_data_manipulate.cu --------------------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//


#include <cooperative_groups.h>
#include <cstdio>
#include <cuda.h>
#include <iostream>
namespace cg = cooperative_groups;



__global__ void test_thread_num_kernel(int *out) {

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);
    cg::thread_block_tile<16> tile16 = cg::tiled_partition<16>(block);
    cg::thread_block_tile<8> tile8 = cg::tiled_partition<8>(block);
    cg::thread_block_tile<4> tile4 = cg::tiled_partition<4>(block);
    cg::thread_block_tile<2> tile2 = cg::tiled_partition<2>(block);
    if (threadIdx.x == 1) {
    // printf("%d\n", tile32.num_threads);
      out[0] = tile32.num_threads();
      out[1] = tile16.num_threads();
      out[2] = tile8.num_threads();
      out[3] = tile4.num_threads();
      out[4] = tile2.num_threads();
      out[5] = block.num_threads();
    }
    
}
bool test_thread_num() {
  int num_elements = 6;
  int expected[] = {32, 16, 8, 4, 2, 56};
  int *output;
  int *result;

  result = (int*)malloc(num_elements * sizeof(int));
  cudaMalloc((void **)&output, num_elements *sizeof(int));
  test_thread_num_kernel<<<1, 56>>>(output);
  cudaMemcpy(result, output, sizeof(unsigned int) * num_elements, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (int i =0; i<num_elements; i++) {
    if (expected[i] != result[i]) {
      return false;
    }
  }
  return true;
}
int main() {
  bool checker4 = test_thread_num();
  if (checker4)
    return 0;
  return -1;
}
