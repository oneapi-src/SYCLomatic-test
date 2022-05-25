// ====------ cooperative_groups.cu --------------------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//


#include <cooperative_groups.h>
#include <cstdio>
namespace cg = cooperative_groups;

// threadIdx.x: 0 ... 7, 8 ... 15, 16 ... 23, 24 ... 31, 32 ... 39, 40 ... 47, 48 ... 55
//              -------  --------  ---------  ---------  ---------  ---------  ---------
//              0        1         2          3          4          5          6

bool verify_array(unsigned int *expected, unsigned int *res, unsigned int size) {
  for (unsigned int i = 0; i < size; i++) {
    if (expected[i] != res[i]) {
      return false;
    }
  }
  return true;
}

__global__ void kernel(unsigned int *data, unsigned int *result) {
  cg::thread_block ttb = cg::this_thread_block();
  cg::thread_block_tile<8> tbt8 = cg::tiled_partition<8>(ttb);

  unsigned int temp = data[threadIdx.x];
  temp = tbt8.shfl_down(temp, 1);
  data[threadIdx.x] = temp;

  if (threadIdx.x == 50) {
    result[0] = tbt8.size();
    result[1] = tbt8.thread_rank();
  }
}

int main() {
  unsigned int result_host[2];
  unsigned int data_host[56];

  for (int i = 0; i < 56; i++) {
    data_host[i] = i;
  }

  unsigned int *result_device, *data_device;
  cudaMalloc(&result_device, sizeof(unsigned int) * 2);
  cudaMalloc(&data_device, sizeof(unsigned int) * 56);

  cudaMemcpy(data_device, &data_host, sizeof(unsigned int) * 56, cudaMemcpyHostToDevice);
  kernel<<<1, 56>>>(data_device, result_device);
  cudaMemcpy(result_host, result_device, sizeof(unsigned int) * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(&data_host, data_device, sizeof(unsigned int) * 56, cudaMemcpyDeviceToHost);
  cudaFree(result_device);
  cudaFree(data_device);

  bool checker1 = false;
  unsigned int expected[56] = {
    1, 2, 3, 4, 5, 6, 7, 7,
    9, 10, 11, 12, 13, 14, 15, 15,
    17, 18, 19, 20, 21, 22, 23, 23,
    25, 26, 27, 28, 29, 30, 31, 31,
    33, 34, 35, 36, 37, 38, 39, 39,
    41, 42, 43, 44, 45, 46, 47, 47,
    49, 50, 51, 52, 53, 54, 55, 55
  };
  if (verify_array(expected, data_host, 56)) {
    checker1 = true;
    printf("checker1 passed\n");
  } else {
    printf("checker1 failed\n");
    for (int i = 0; i < 7; i++) {
      for (int j = 0; j < 8; j++) {
        int idx = i * 8 + j;
        printf("%d, ", data_host[idx]);
      }
      printf("\n");
    }
  }

  bool checker2 = false;
  if (result_host[0] == 8 &&
      result_host[1] == 2) {
    checker2 = true;
    printf("checker2 passed\n");
  } else {
    printf("checker2 failed\n");
    printf("%d, %d\n", result_host[0], result_host[1]);
  }

  if (checker1 && checker2)
    return 0;
  return -1;
}
