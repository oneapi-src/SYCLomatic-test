// ====------ cub_device.cu------------------------------ *- CUDA -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <algorithm>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define DATA_NUM 100

template<typename T = int>
T *init_data(std::initializer_list<T> init) {
  T *Ptr = nullptr;
  cudaMallocManaged(&Ptr, sizeof(T) * init.size());
  memcpy(Ptr, init.begin(), sizeof(T) * init.size());
  return Ptr;
}

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

std::ostream &operator<<(std::ostream &os, const cub::KeyValuePair<int, int> &kv) {
  os << '[' << kv.key << ", " << kv.value << ']';
  return os;
}

bool test_arg_min() {
  int num_segs = 3;
  int *offset = init_data({0, 3, 3, 7});
  int *in = init_data({8, 6, 7, 5, 3, 0, 9});

  cub::KeyValuePair<int, int> *out = init_data<cub::KeyValuePair<int, int>>({{}, {}, {}});
  cub::KeyValuePair<int, int> expected[] = {{1, 6}, {1, INT_MAX}, {2, 0}};

  // CHECK-DPCT1026 DPCT1026:{{.*}}: The call to cub::DeviceSegmentedReduce::ArgMin was removed because this call is redundant in SYCL.
  // CHECK: dpct::segmented_reduce_argmin(oneapi::dpl::execution::device_policy(q_ct1), in, out, num_segs, offset, offset + 1);
  void *tmp_storage = nullptr;
  size_t tmp_storage_size = 0;
  cub::DeviceSegmentedReduce::ArgMin(tmp_storage, tmp_storage_size, in, out, num_segs, offset, offset + 1);
  cudaMalloc(&tmp_storage, tmp_storage_size);
  cub::DeviceSegmentedReduce::ArgMin(tmp_storage, tmp_storage_size, in, out, num_segs, offset, offset + 1);
  cudaDeviceSynchronize();

  auto cmp = [](const cub::KeyValuePair<int, int> &lhs, const cub::KeyValuePair<int, int> &rhs) -> bool {
    return lhs.value == rhs.value && lhs.key == rhs.key;
  };

  if (!std::equal(out, out + num_segs, expected, cmp)) {
    std::cout << "ArgMin verify failed!\n";
    std::cout << "expect: ";
    std::for_each(expected, expected + num_segs, [](const auto &v) { std::cout << v << " "; });
    std::cout << "\n";
    std::cout<< "current result: ";
    std::for_each(expected, expected + num_segs, [](const auto &v) { std::cout << v << " "; });
    std::cout << "\n";
    return false;
  }
  return true;
}

bool test_arg_max() {
  int num_segs = 3;
  int *offset = init_data({0, 3, 3, 7});
  int *in = init_data({8, 6, 7, 5, 3, 0, 9});

  cub::KeyValuePair<int, int> *out = init_data<cub::KeyValuePair<int, int>>({{}, {}, {}});
  cub::KeyValuePair<int, int> expected[] = {{0, 8}, {1, INT_MIN}, {3, 9}};

  // CHECK-DPCT1026 DPCT1026:{{.*}}: The call to cub::DeviceSegmentedReduce::ArgMax was removed because this call is redundant in SYCL.
  // CHECK: dpct::segmented_reduce_argmax(oneapi::dpl::execution::device_policy(q_ct1), in, out, num_segs, offset, offset + 1);
  void *tmp_storage = nullptr;
  size_t tmp_storage_size = 0;
  cub::DeviceSegmentedReduce::ArgMax(tmp_storage, tmp_storage_size, in, out, num_segs, offset, offset + 1);
  cudaMalloc(&tmp_storage, tmp_storage_size);
  cub::DeviceSegmentedReduce::ArgMax(tmp_storage, tmp_storage_size, in, out, num_segs, offset, offset + 1);
  cudaDeviceSynchronize();

  auto cmp = [](const cub::KeyValuePair<int, int> &lhs, const cub::KeyValuePair<int, int> &rhs) -> bool {
    return lhs.value == rhs.value && lhs.key == rhs.key;
  };

  if (!std::equal(out, out + num_segs, expected, cmp)) {
    std::cout << "ArgMax verify failed!\n";
    std::cout << "expect: ";
    std::for_each(expected, expected + num_segs, [](const auto &v) { std::cout << v << " "; });
    std::cout << "\n";
    std::cout<< "current result: ";
    std::for_each(expected, expected + num_segs, [](const auto &v) { std::cout << v << " "; });
    std::cout << "\n";
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
  Result = test_arg_min() && Result;
  Result = test_arg_max() && Result;
  if(Result) {
    std::cout << "cub_device Pass" << std::endl;
    return 0;
  }
  return 1;
}

