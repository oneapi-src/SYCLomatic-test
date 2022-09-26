// ====------ cub_device_reduce_sum.cu-------------------- *- CUDA -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//


#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define DATA_NUM 100

template<typename T = int>
void init_data(T* data, int num) {
  T host_data[DATA_NUM];
  for(int i = 0; i < num; i++)
    host_data[i] = i;
  cudaMemcpy(data, host_data, num * sizeof(T), cudaMemcpyHostToDevice);
}
template<typename T = int>
bool verify_data(T* data, T* expect, int num, int step = 1) {
  T host_data[DATA_NUM];
  cudaMemcpy(host_data, data, num * sizeof(T), cudaMemcpyDeviceToHost);
  for(int i = 0; i < num; i = i + step) {
    if(host_data[i] != expect[i]) {
      return false;
    }
  }
  return true;
}
template<typename T = int>
void print_data(T* data, int num, bool IsHost = false) {
  if(IsHost) {
    for (int i = 0; i < num; i++) {
      std::cout << data[i] << ", ";
      if((i+1)%32 == 0)
        std::cout << std::endl;
    }
    std::cout << std::endl;
    return;
  }
  T host_data[DATA_NUM];
  cudaMemcpy(host_data, data, num * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < num; i++) {
    std::cout << host_data[i] << ", ";
    if((i+1)%32 == 0)
        std::cout << std::endl;
  }
  std::cout << std::endl;
}

/// cub::DeviceReduce::Sum
bool test_device_reduce_sum() {
  int *device_in;
  int *device_out;
  void *temp_storage = NULL;
  size_t temp_storage_size = 0;
  int expect = 4950;
  cudaMalloc((void **)&device_in, sizeof(int) * DATA_NUM);
  cudaMalloc((void **)&device_out, sizeof(int));
  init_data(device_in, DATA_NUM);
  cub::DeviceReduce::Sum(temp_storage, temp_storage_size, device_in,
                         device_out, DATA_NUM);
  cudaMalloc((void **)&temp_storage, temp_storage_size);
  cub::DeviceReduce::Sum(temp_storage, temp_storage_size, device_in,
                         device_out, DATA_NUM);
  cudaDeviceSynchronize();
  if (!verify_data(device_out, &expect, 1)) {
    std::cout << "cub::DeviceReduce::Sum verify failed\n";
    std::cout << "expect:\n";
    print_data<int>(&expect, 1, true);
    std::cout << "current result:\n";
    print_data<int>(device_out, 1);
    return false;
  }
  return true;
}

int main() {
  if (test_device_reduce_sum()) {
    std::cout << "cub::DeviceReduce::Sum Pass\n";
    return 0;
  }
  return 1;
}
