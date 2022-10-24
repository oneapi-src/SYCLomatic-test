// ====------ cub_device.cu------------------------------ *- CUDA -* ------===//
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

bool test_reduce_1(){
  int          num_segments = 10;
  int          *device_offsets;
  int          *device_in;
  int          *device_out;
  int          initial_value = INT_MAX;
  void     *temp_storage = NULL;
  size_t   temp_storage_size = 0;
  int expect[DATA_NUM] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};

  cudaMalloc(&device_offsets, (num_segments + 1) * sizeof(int));
  cudaMalloc(&device_in, DATA_NUM * sizeof(int));
  cudaMalloc(&device_out, num_segments * sizeof(int));
  init_data(device_in, DATA_NUM);
  int host_offsets[11];
  for(int i = 0; i < num_segments + 1; i++) {
    host_offsets[i] = i * 10;
  }
  cudaMemcpy(device_offsets, host_offsets, 11 * sizeof(int), cudaMemcpyHostToDevice);

  cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1, cub::Min(), initial_value);

  cudaMalloc(&temp_storage, temp_storage_size);

  cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1, cub::Min(), initial_value);

  cudaDeviceSynchronize();

  if(!verify_data(device_out, expect, num_segments)) {
    std::cout << "Reduce" << " verify failed" << std::endl;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect, num_segments, true);
    std::cout << "current result:" << std::endl;
    print_data<int>(device_out, num_segments);
    return false;
  }
  return true;
}


bool test_sum_1(){
  int          num_segments = 10;
  int          *device_offsets;
  int          *device_in;
  int          *device_out;
  void     *temp_storage = NULL;
  size_t   temp_storage_size = 0;
  int expect[DATA_NUM] = {45, 145, 245, 345, 445, 545, 645, 745, 845, 945};

  cudaMalloc(&device_offsets, (num_segments + 1) * sizeof(int));
  cudaMalloc(&device_in, DATA_NUM * sizeof(int));
  cudaMalloc(&device_out, num_segments * sizeof(int));
  init_data(device_in, DATA_NUM);
  int host_offsets[11];
  for(int i = 0; i < num_segments + 1; i++) {
    host_offsets[i] = i * 10;
  }
  cudaMemcpy(device_offsets, host_offsets, 11 * sizeof(int), cudaMemcpyHostToDevice);

  cub::DeviceSegmentedReduce::Sum(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1);

  cudaMalloc(&temp_storage, temp_storage_size);

  cub::DeviceSegmentedReduce::Sum(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1);

  cudaDeviceSynchronize();

  if(!verify_data(device_out, expect, num_segments)) {
    std::cout << "Sum" << " verify failed" << std::endl;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect, num_segments, true);
    std::cout << "current result:" << std::endl;
    print_data<int>(device_out, num_segments);
    return false;
  }
  return true;
}

bool test_sum_2(){
  int          num_segments = 10;
  int          *device_offsets;
  int          *device_in;
  int          *device_out;
  void     *temp_storage = NULL;
  size_t   temp_storage_size = 0;
  int expect[DATA_NUM] = {190, 0, 245, 345, 445, 545, 645, 745, 845, 945};

  cudaMalloc(&device_offsets, (num_segments + 1) * sizeof(int));
  cudaMalloc(&device_in, DATA_NUM * sizeof(int));
  cudaMalloc(&device_out, num_segments * sizeof(int));
  init_data(device_in, DATA_NUM);
  int host_offsets[11];
  for(int i = 0; i < num_segments + 1; i++) {
    host_offsets[i] = i * 10;
  }
  host_offsets[1] = 20;
  cudaMemcpy(device_offsets, host_offsets, 11 * sizeof(int), cudaMemcpyHostToDevice);
  cub::DeviceSegmentedReduce::Sum(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1);

  cudaMalloc(&temp_storage, temp_storage_size);

  cub::DeviceSegmentedReduce::Sum(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1);

  cudaDeviceSynchronize();

  if(!verify_data(device_out, expect, num_segments)) {
    std::cout << "Sum" << " verify failed" << std::endl;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect, num_segments, true);
    std::cout << "current result:" << std::endl;
    print_data<int>(device_out, num_segments);
    return false;
  }
  return true;
}

bool test_min(){
  int          num_segments = 10;
  int          *device_offsets;
  int          *device_in;
  int          *device_out;
  void     *temp_storage = NULL;
  size_t   temp_storage_size = 0;
  int expect[DATA_NUM] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};

  cudaMalloc(&device_offsets, (num_segments + 1) * sizeof(int));
  cudaMalloc(&device_in, DATA_NUM * sizeof(int));
  cudaMalloc(&device_out, num_segments * sizeof(int));
  init_data(device_in, DATA_NUM);
  int host_offsets[11];
  for(int i = 0; i < num_segments + 1; i++) {
    host_offsets[i] = i * 10;
  }
  cudaMemcpy(device_offsets, host_offsets, 11 * sizeof(int), cudaMemcpyHostToDevice);

  cub::DeviceSegmentedReduce::Min(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1);

  cudaMalloc(&temp_storage, temp_storage_size);

  cub::DeviceSegmentedReduce::Min(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1);

  cudaDeviceSynchronize();

  if(!verify_data(device_out, expect, num_segments)) {
    std::cout << "Min" << " verify failed" << std::endl;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect, num_segments, true);
    std::cout << "current result:" << std::endl;
    print_data<int>(device_out, num_segments);
    return false;
  }
  return true;
}


bool test_max(){
  int          num_segments = 10;
  int          *device_offsets;
  int          *device_in;
  int          *device_out;
  void     *temp_storage = NULL;
  size_t   temp_storage_size = 0;
  int expect[DATA_NUM] = {9, 19, 29, 39, 49, 59, 69, 79, 89, 99};

  cudaMalloc(&device_offsets, (num_segments + 1) * sizeof(int));
  cudaMalloc(&device_in, DATA_NUM * sizeof(int));
  cudaMalloc(&device_out, num_segments * sizeof(int));
  init_data(device_in, DATA_NUM);

  int host_offsets[11];
  for(int i = 0; i < num_segments + 1; i++) {
    host_offsets[i] = i * 10;
  }
  cudaMemcpy(device_offsets, host_offsets, 11 * sizeof(int), cudaMemcpyHostToDevice);

  cub::DeviceSegmentedReduce::Max(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1);

  cudaMalloc(&temp_storage, temp_storage_size);

  cub::DeviceSegmentedReduce::Max(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1);

  cudaDeviceSynchronize();

  if(!verify_data(device_out, expect, num_segments)) {
    std::cout << "Max" << " verify failed" << std::endl;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect, num_segments, true);
    std::cout << "current result:" << std::endl;
    print_data<int>(device_out, num_segments);
    return false;
  }
  return true;
}

int main() {
  bool Result = true;
  Result = test_reduce_1() && Result;
  Result = test_sum_1() && Result;
  Result = test_sum_2() && Result;
  Result = test_min() && Result;
  Result = test_max() && Result;
  if(Result) {
    std::cout << "cub_device Pass" << std::endl;
    return 0;
  }
  return 1;
}

