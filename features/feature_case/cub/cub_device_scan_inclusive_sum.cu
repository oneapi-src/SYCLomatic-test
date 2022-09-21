// ====------ cub_device_scan_inclusive_sum.cu------------ *- CUDA -* ------===//
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

// cub::DeviceScan::InclusiveSum
bool test_device_scan_inclusive_sum() {
  static const int n = 10;
  int *device_in;
  int *device_out;
  void *temp_storage = NULL;
  size_t temp_storage_size = 0;
  int expect[n] = {0, 1, 3, 6, 10, 15, 21, 28, 36, 45};
  cudaMalloc((void **)&device_in, sizeof(int) * n);
  cudaMalloc((void **)&device_out, sizeof(int) * n);
  init_data(device_in, n);
  cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_size, device_in,
                                device_out, n);
  cudaMalloc((void **)&temp_storage, temp_storage_size);
  cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_size, device_in,
                                device_out, n);
  cudaDeviceSynchronize();
  if (!verify_data(device_out, expect, n)) {
    std::cout << "cub::DeviceScan::InclusiveSum verify failed\n";
    std::cout << "expect:\n";
    print_data<int>(expect, 1, true);
    std::cout << "current result:\n";
    print_data<int>(device_out, n);
    return false;
  }
  return true;
}

int main() {
  if (test_device_scan_inclusive_sum())
    std::cout << "cub::DeviceScan::InclusiveSum Pass\n";
  return 0;
}
