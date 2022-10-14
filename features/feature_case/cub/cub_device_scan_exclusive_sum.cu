// ====------ cub_device_scan_exclusive_sum.cu------------ *- CUDA -* ------===//
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
#define EPS (1e-6)

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

// cub::DeviceScan::ExclusiveSum
bool test_device_scan_exclusive_sum() {
  static const int n = 10;
  int *device_in;
  int *device_out;
  void *temp_storage = NULL;
  size_t temp_storage_size = 0;
  int expect[n] = {0, 0, 1, 3, 6, 10, 15, 21, 28, 36};
  cudaMalloc((void **)&device_in, sizeof(int) * n);
  cudaMalloc((void **)&device_out, sizeof(int) * n);
  init_data(device_in, n);
  cub::DeviceScan::ExclusiveSum(temp_storage, temp_storage_size, device_in,
                                device_out, n);
  cudaMalloc((void **)&temp_storage, temp_storage_size);
  cub::DeviceScan::ExclusiveSum(temp_storage, temp_storage_size, device_in,
                                device_out, n);
  cudaDeviceSynchronize();
  if (!verify_data(device_out, expect, n)) {
    std::cout << "cub::DeviceScan::ExclusiveSum verify failed\n";
    std::cout << "expect:\n";
    print_data<int>(expect, 1, true);
    std::cout << "current result:\n";
    print_data<int>(device_out, n);
    return false;
  }
  return true;
}

bool test_device_scan_exclusive_sum2() {
  static constexpr int num_items = 4;
  float ret[num_items];
  float f4[] = {0.1, 0.2, 0.3, 0.4};
  float expect[] = {0.0, 0.1, 0.3, 0.6};
  float *d_in;
  float *d_out;
  void *tmp = nullptr;
  size_t tmp_size = 0;
  cudaMalloc((void **)&d_in, num_items * sizeof(float));
  cudaMalloc((void **)&d_out, num_items * sizeof(float));
  cudaMemcpy(d_in, f4, sizeof(f4), cudaMemcpyHostToDevice);
  cub::DeviceScan::ExclusiveSum(tmp, tmp_size, d_in, d_out, num_items);
  cudaMalloc((void **)&tmp, tmp_size);
  cub::DeviceScan::ExclusiveSum(tmp, tmp_size, d_in, d_out, num_items);
  cudaMemcpy(ret, d_out, num_items * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < num_items; ++i) {
    float eps = ret[i] - expect[i];
    if (eps < -EPS || eps > EPS)
      return false;
  }
  return true;
}

int main() {
  if (test_device_scan_exclusive_sum() && test_device_scan_exclusive_sum2()) {
    std::cout << "cub::DeviceScan::ExclusiveSum Pass\n";
    return 0;
  }
  return 1;
}
