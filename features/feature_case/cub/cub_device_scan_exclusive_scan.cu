// ====------ cub_device_scan_exclusive_scan.cu----------- *- CUDA -* ------===//
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

struct ScanOp {
  template <typename T, typename = typename std::enable_if<
                            std::is_arithmetic<T>::value>::type>
  __device__ T operator()(const T &lhs, const T &rhs) const {
    return lhs + rhs;
  }
};

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

// cub::DeviceScan::ExclusiveScan
bool test_device_exclusive_scan() {
  static const int n = 10;
  int *device_in;
  int *device_out;
  void *temp_storage = NULL;
  size_t temp_storage_size = 0;
  int expect[n] = {0, 0, 1, 3, 6, 10, 15, 21, 28, 36};
  cudaMalloc((void **)&device_in, sizeof(int) * n);
  cudaMalloc((void **)&device_out, sizeof(int) * n);
  init_data(device_in, n);
  ScanOp scan_op;
  cub::DeviceScan::ExclusiveScan(temp_storage, temp_storage_size, device_in,
                                device_out, scan_op, 0, n);
  cudaMalloc((void **)&temp_storage, temp_storage_size);
  cub::DeviceScan::ExclusiveScan(temp_storage, temp_storage_size, device_in,
                                device_out, scan_op, 0, n);
  cudaDeviceSynchronize();
  if (!verify_data(device_out, expect, n)) {
    std::cout << "cub::DeviceScan::ExclusiveScan verify failed\n";
    std::cout << "expect:\n";
    print_data<int>(expect, n, true);
    std::cout << "current result:\n";
    print_data<int>(device_out, n);
    return false;
  }
  return true;
}


int main() {
  if (test_device_exclusive_scan())
    std::cout << "cub::DeviceScan::ExclusiveScan Pass\n";
  return 0;
}
