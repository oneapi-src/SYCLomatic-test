// ====------ cooperative_groups_thread_group.cu --------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cooperative_groups.h>
#include <cstdio>
#include <stdlib.h>
namespace cg = cooperative_groups;

__device__ int testThreadGroup(cg::thread_group g, int *input, int val) {

  int thread_index = g.thread_rank();
  for (int i = g.size() / 2; i > 0; i /= 2) {
    input[thread_index] = val;
    g.sync();

    if (thread_index < i) {
      val += input[thread_index];
    }
    g.sync();
  }
  if (thread_index == 0) {
    return val;
  } else {
    return -1;
  }
}

__global__ void kernelFunc(unsigned int *ret) {
  *ret = 0;
  auto block = cg::this_thread_block();
  int value = 2;
  __shared__ int workspace[1024];
  block.thread_index();
  auto threadBlockGroup = cg::this_thread_block();
  int ret1, ret2, ret3;
  ret1 = testThreadGroup(threadBlockGroup, workspace, value);
  if (threadBlockGroup.thread_rank() == 0) {
    printf("value1 is %d\n", ret1);
  }

  cg::thread_block_tile<16> tilePartition16 =
      cg::tiled_partition<16>(threadBlockGroup);
  ret2 = testThreadGroup(tilePartition16, workspace, value);
  if (threadBlockGroup.thread_rank() == 0) {
    printf("value2 is %d\n", ret2);
  }

  cg::thread_block_tile<32> tilePartition32 =
      cg::tiled_partition<32>(threadBlockGroup);
  ret3 = testThreadGroup(tilePartition32, workspace, value);
  if (threadBlockGroup.thread_rank() == 0) {
    printf("value3 is %d\n", ret3);
  }
  if (threadBlockGroup.thread_rank() == 0) {
    if (ret1 == 512 && ret2 == 32 && ret3 == 64) {
      *ret = 1;
    } else {
      *ret = -1;
    }
  }
}

int main() {
  bool checker4 = false;
  unsigned int *ret_result;
  unsigned int host[1];
  cudaMalloc(&ret_result, sizeof(unsigned int));
  kernelFunc<<<1, 256>>>(ret_result);
  cudaMemcpy(host, ret_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaFree(ret_result);
  printf("host valu is %d \n ", host[0]);
  if (host[0] == 1) {
    printf(" thread_group migration is run success \n");
    checker4 = true;
  } else {
    printf("thread_group migration is run failed\n ");
  }

  if (checker4)
    return 0;
  return -1;
}
