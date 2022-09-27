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
#include <algorithm>
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
bool verify_data(T* data, T* expect, int num) {
  T host_data[DATA_NUM];
  cudaMemcpy(host_data, data, num * sizeof(T), cudaMemcpyDeviceToHost);
  for(int i = 0; i < num; ++i) {
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

bool test_device_reduce(void) {
  size_t n_tmp;
  CustomMin op;
  static const int n = 7;
  int in[] = {8, 6, 7, 5, -1, 0, 9};
  int *d_in, *d_out, *tmp = nullptr, expect_ans = *std::min_element(in, in + n);
  cudaMalloc((void **)&d_in, sizeof(int) * n);
  cudaMalloc((void **)&d_out, sizeof(int) * n);
  cudaMemcpy((void *)d_in, (void *)in, sizeof(in), cudaMemcpyHostToDevice);
  cub::DeviceReduce::Reduce(tmp, n_tmp, d_in, d_out, n, op, 0);
  cudaMalloc((void **)&tmp, n_tmp);
  cub::DeviceReduce::Reduce(tmp, n_tmp, d_in, d_out, n, op, 0);
  cudaDeviceSynchronize();
  if (!verify_data(d_out, &expect_ans, 1)) {
    std::cout << "cub::DeviceReduce::Reduce verify failed\n";
    std::cout << "expect:\n";
    print_data<int>(&expect_ans, 1, true);
    std::cout << "current result:\n";
    print_data<int>(d_out, 1);
    return false;
  }
  return true;
}

int main() {
  if (test_device_reduce()) {
    std::cout << "cub::DeviceReduce::Reduce Pass\n";
    return 0;
  }
  return 1;
}
