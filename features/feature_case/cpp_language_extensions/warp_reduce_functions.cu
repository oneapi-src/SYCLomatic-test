// ===----------- warp_reduce_functions.cu---------- *- CUDA -* -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define DATA_NUM 128

void print_data(unsigned *data) {
  std::vector<unsigned> host_data(DATA_NUM);
  cudaMemcpy(host_data.data(), data, DATA_NUM * sizeof(unsigned),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < DATA_NUM; i++) {
    std::cout << host_data[i] << ", ";
    if ((i + 1) % 32 == 0)
      std::cout << std::endl;
  }
  std::cout << std::endl;
}

bool check_data(unsigned *data, int first, int second, int third, int fourth) {
  std::vector<unsigned> host_data(DATA_NUM);
  cudaMemcpy(host_data.data(), data, DATA_NUM * sizeof(unsigned),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < DATA_NUM / 4; i++) {
    if (host_data[i] != first)
      return false;
  }
  for (int i = DATA_NUM / 4; i < DATA_NUM / 2; i++) {
    if (host_data[i] != second)
      return false;
  }
  for (int i = DATA_NUM / 2; i < DATA_NUM * 3 / 4; i++) {
    if (host_data[i] != third)
      return false;
  }
  for (int i = DATA_NUM * 3 / 4; i < DATA_NUM; i++) {
    if (host_data[i] != fourth)
      return false;
  }
  return true;
}

__global__ void reduce_add_sync(unsigned int *data) {
  int thread_id = threadIdx.x + threadIdx.y * blockDim.x +
                  threadIdx.z * blockDim.x * blockDim.y +
                  blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  data[thread_id] = __reduce_add_sync(0xFFFF, thread_id);
}

__global__ void reduce_min_sync(unsigned int *data) {
  int thread_id = threadIdx.x + threadIdx.y * blockDim.x +
                  threadIdx.z * blockDim.x * blockDim.y +
                  blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  data[thread_id] = __reduce_min_sync(0xFFFF, thread_id);
}

__global__ void reduce_max_sync(unsigned int *data) {
  int thread_id = threadIdx.x + threadIdx.y * blockDim.x +
                  threadIdx.z * blockDim.x * blockDim.y +
                  blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  data[thread_id] = __reduce_max_sync(0xFFFF, thread_id);
}

__global__ void reduce_and_sync(unsigned int *data) {
  int thread_id = threadIdx.x + threadIdx.y * blockDim.x +
                  threadIdx.z * blockDim.x * blockDim.y +
                  blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  data[thread_id] = __reduce_and_sync(0xFFFF, thread_id);
}

__global__ void reduce_or_sync(unsigned int *data) {
  int thread_id = threadIdx.x + threadIdx.y * blockDim.x +
                  threadIdx.z * blockDim.x * blockDim.y +
                  blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  data[thread_id] = __reduce_or_sync(0xFFFF, thread_id);
}

__global__ void reduce_xor_sync(unsigned int *data) {
  int thread_id = threadIdx.x + threadIdx.y * blockDim.x +
                  threadIdx.z * blockDim.x * blockDim.y +
                  blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  data[thread_id] = __reduce_xor_sync(0xFFFF, thread_id);
}

int main() {
  int ret = 0;
  unsigned *dev_data = nullptr;
  cudaMalloc(&dev_data, DATA_NUM * sizeof(unsigned int));

  reduce_add_sync<<<2, 64>>>(dev_data);
  cudaDeviceSynchronize();
  if (!check_data(dev_data, 496, 1520, 2544, 3568)) {
    print_data(dev_data);
    std::cout << "reduce_add_sync check failed!" << std::endl;
    ret++;
  } else {
    std::cout << "reduce_add_sync check passed!" << std::endl;
  }

  reduce_min_sync<<<2, 64>>>(dev_data);
  cudaDeviceSynchronize();
  if (!check_data(dev_data, 0, 32, 64, 96)) {
    print_data(dev_data);
    std::cout << "reduce_min_sync check failed!" << std::endl;
    ret++;
  } else {
    std::cout << "reduce_min_sync check passed!" << std::endl;
  }

  reduce_max_sync<<<2, 64>>>(dev_data);
  cudaDeviceSynchronize();
  if (!check_data(dev_data, 31, 63, 95, 127)) {
    print_data(dev_data);
    std::cout << "reduce_max_sync check failed!" << std::endl;
    ret++;
  } else {
    std::cout << "reduce_max_sync check passed!" << std::endl;
  }

  reduce_and_sync<<<2, 64>>>(dev_data);
  cudaDeviceSynchronize();
  if (!check_data(dev_data, 0, 32, 64, 96)) {
    print_data(dev_data);
    std::cout << "reduce_and_sync check failed!" << std::endl;
    ret++;
  } else {
    std::cout << "reduce_and_sync check passed!" << std::endl;
  }

  reduce_or_sync<<<2, 64>>>(dev_data);
  cudaDeviceSynchronize();
  if (!check_data(dev_data, 31, 63, 95, 127)) {
    print_data(dev_data);
    std::cout << "reduce_or_sync check failed!" << std::endl;
    ret++;
  } else {
    std::cout << "reduce_or_sync check passed!" << std::endl;
  }

  reduce_xor_sync<<<2, 64>>>(dev_data);
  cudaDeviceSynchronize();
  if (!check_data(dev_data, 0, 0, 0, 0)) {
    print_data(dev_data);
    std::cout << "reduce_xor_sync check failed!" << std::endl;
    ret++;
  } else {
    std::cout << "reduce_xor_sync check passed!" << std::endl;
  }

  return ret;
}
