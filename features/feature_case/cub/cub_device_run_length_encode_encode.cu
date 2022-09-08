// ====------ cub_device_run_length_encode_encode.cu------- *- CUDA -* ------===//
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

// cub::DeviceRunLengthEncode::Encode
bool test_device_encode() {
  static const int N = 8;
  int data[N] = {0, 2, 2, 9, 5, 5, 5, 8};
  int *d_in = nullptr;
  int *d_temp = nullptr;
  int *d_unique = nullptr;
  int *d_counts = nullptr;
  int *d_selected_num = nullptr;
  int h_selected_num = 0;
  size_t d_temp_size = 0;

  cudaMalloc((void **)&d_in, sizeof(int) * N);
  cudaMalloc((void **)&d_unique, sizeof(int) * N);
  cudaMalloc((void **)&d_counts, sizeof(int) * N);
  cudaMalloc((void **)&d_selected_num, sizeof(int));
  cudaMemcpy((void *)d_in, (void *)data, sizeof(data), cudaMemcpyHostToDevice);
  cub::DeviceRunLengthEncode::Encode(nullptr, d_temp_size, d_in, d_unique, d_counts, d_selected_num, N);
  cudaMalloc((void **)&d_temp, d_temp_size);
  cub::DeviceRunLengthEncode::Encode(d_temp, d_temp_size, d_in, d_unique, d_counts, d_selected_num, N);
  cudaDeviceSynchronize();

  int expect_select_num = 5;
  int expect_unique[] = {0, 2, 9, 5, 8};
  int expect_counts[] = {1, 2, 1, 3, 1};

  
  if (!verify_data(d_selected_num, &expect_select_num, 1)) {
    std::cout << "cub::DeviceRunLengthEncode::Encode select_num verify failed\n";
    std::cout << "expect:\n";
    print_data<int>(&expect_select_num, 1, true);
    std::cout << "current result:\n";
    print_data<int>(d_selected_num, 1);
    return false;
  }

  if (!verify_data(d_unique, (int *)expect_unique, expect_select_num)) {
    std::cout << "cub::DeviceRunLengthEncode::Encode output unique data verify failed\n";
    std::cout << "expect:\n";
    print_data<int>(expect_unique, 1, true);
    std::cout << "current result:\n";
    print_data<int>(d_unique, 1);
    return false;
  }

   if (!verify_data(d_counts, (int *)expect_counts, expect_select_num)) {
    std::cout << "cub::DeviceRunLengthEncode::Encode output counts data verify failed\n";
    std::cout << "expect:\n";
    print_data<int>(expect_counts, 1, true);
    std::cout << "current result:\n";
    print_data<int>(d_counts, 1);
    return false;
  }

  return true;
}

int main() {
  if (test_device_encode())
    std::cout << "cub::DeviceRunLengthEncode::Encode Pass\n";
  return 0;
}
