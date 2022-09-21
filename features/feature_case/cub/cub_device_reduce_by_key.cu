// ====------ cub_device_reduce.cu------------------------ *- CUDA -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <vector>

#define DATA_NUM 100

struct CustomMin {
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return (b < a) ? b : a;
    }
};

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

bool test_device_reduce_by_key(void) {
   int n = 8;
  CustomMin op;
  int key[] = {0, 2, 2, 9, 5, 5, 5, 8};
  int val[] = {0, 7, 1, 6, 2, 5, 3, 4};
  int expect_unq[] = {0, 2, 9, 5, 8};
  int expect_agg[] = {0, 1, 6, 2, 4};
  int *d_key, *d_val, *d_unq, *d_agg, *d_num, expect_num = 5;
  cudaMalloc(&d_key, sizeof(key));
  cudaMalloc(&d_val, sizeof(val));
  cudaMalloc(&d_unq, sizeof(key));
  cudaMalloc(&d_agg, sizeof(key));
  cudaMalloc(&d_num, sizeof(int));
  cudaMemcpy(d_key, key, sizeof(key), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  cudaMemcpy(d_val, val, sizeof(val), cudaMemcpyHostToDevice);

  void *tmp = nullptr;
  size_t n_tmp = 0;
  cub::DeviceReduce::ReduceByKey(tmp, n_tmp, d_key, d_unq, d_val, d_agg, d_num, op, n);
  cudaMalloc(&tmp, n_tmp);
  cub::DeviceReduce::ReduceByKey(tmp, n_tmp, d_key, d_unq, d_val, d_agg, d_num, op, n);

  cudaDeviceSynchronize();

  if (!verify_data(d_num, &expect_num, 1)) {
    std::cout << "cub::DeviceReduce::ReduceByKey select_num verify failed\n";
    std::cout << "expect:\n";
    print_data<int>(&expect_num, 1, true);
    std::cout << "current result:\n";
    print_data<int>(d_num, 1);
    return false;
  }

  if (!verify_data(d_unq, expect_unq, expect_num)) {
    std::cout << "cub::DeviceReduce::ReduceByKey output data verify failed\n";
    std::cout << "expect:\n";
    print_data<int>(expect_unq, expect_num, true);
    std::cout << "current result:\n";
    print_data<int>(d_unq, expect_num);
    return false;
  }

  if (!verify_data(d_agg, expect_agg, expect_num)) {
    std::cout << "cub::DeviceReduce::ReduceByKey output data verify failed\n";
    std::cout << "expect:\n";
    print_data<int>(expect_agg, expect_num, true);
    std::cout << "current result:\n";
    print_data<int>(d_agg, expect_num);
    return false;
  }

  return true;
}

int main() {
  if (test_device_reduce_by_key()) {
    std::cout << "cub::DeviceReduce::ReduceByKey Pass\n";
    return 0;
  }
  return 1;
}
