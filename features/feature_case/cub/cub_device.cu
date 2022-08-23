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
  int host_offsets[10];
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
  int host_offsets[10];
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
  int host_offsets[10];
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
  int host_offsets[10];
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

  int host_offsets[10];
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

// cub::DeviceSelect::Flagged
bool test_device_select_flagged() {
  static const int n = 10;
  int *device_in = nullptr;
  int *device_out = nullptr;
  int *device_flagged = nullptr;
  int *device_select_num = nullptr;
  int *device_tmp = nullptr;
  size_t n_device_tmp = 0;
  int host_in[n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int host_flagged[n] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  int expect_out[] = {1, 3, 5, 7, 9};
  int expect_select_num = 5;
  cudaMalloc((void **)&device_in, n * sizeof(int));
  cudaMalloc((void **)&device_out, n * sizeof(int));
  cudaMalloc((void **)&device_flagged, n * sizeof(int));
  cudaMalloc((void **)&device_select_num, sizeof(int));
  cudaMemcpy(device_in, (void *)host_in, sizeof(host_in),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_flagged, host_flagged, sizeof(host_flagged),
             cudaMemcpyHostToDevice);
  cub::DeviceSelect::Flagged(device_tmp, n_device_tmp, device_in,
                             device_flagged, device_out, device_select_num,
                             n);
  cudaMalloc((void **)&device_tmp, n_device_tmp);
  cub::DeviceSelect::Flagged(device_tmp, n_device_tmp, device_in,
                             device_flagged, device_out, device_select_num,
                             n);
  cudaDeviceSynchronize();

  if (!verify_data(device_select_num, &expect_select_num, 1)) {
    std::cout << "cub::DeviceSelect::Flagged select_num verify failed\n";
    std::cout << "expect:\n";
    print_data<int>(&expect_select_num, 1, true);
    std::cout << "current result:\n";
    print_data<int>(device_select_num, 1);
    return false;
  }

  if (!verify_data(device_out, (int *)expect_out, expect_select_num)) {
    std::cout << "cub::DeviceSelect::Flagged output data verify failed\n";
    std::cout << "expect:\n";
    print_data<int>(expect_out, 1, true);
    std::cout << "current result:\n";
    print_data<int>(device_out, 1);
    return false;
  }
  return true;
}

// cub::DeviceScan::InclusiveScan
bool test_device_inclusive_scan() {
  static const int n = 10;
  int *device_in;
  int *device_out;
  void *temp_storage = NULL;
  size_t temp_storage_size = 0;
  int expect[n] = {0, 1, 3, 6, 10, 15, 21, 28, 36, 45};
  cudaMalloc((void **)&device_in, sizeof(int) * n);
  cudaMalloc((void **)&device_out, sizeof(int) * n);
  init_data(device_in, n);
  ScanOp scan_op;
  cub::DeviceScan::InclusiveScan(temp_storage, temp_storage_size, device_in,
                                device_out, scan_op, n);
  cudaMalloc((void **)&temp_storage, temp_storage_size);
  cub::DeviceScan::InclusiveScan(temp_storage, temp_storage_size, device_in,
                                device_out, scan_op, n);
  cudaDeviceSynchronize();
  if (!verify_data(device_out, expect, n)) {
    std::cout << "cub::DeviceScan::InclusiveScan verify failed\n";
    std::cout << "expect:\n";
    print_data<int>(expect, 1, true);
    std::cout << "current result:\n";
    print_data<int>(device_out, n);
    return false;
  }
  return true;
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

// cub::DeviceSelect::Unique
bool test_device_unique() {
  static const int N = 8;
  int data[N] = {0, 2, 2, 9, 5, 5, 5, 8};
  int *d_in = nullptr;
  int *d_out = nullptr;
  int *d_temp = nullptr;
  int *d_selected_num = nullptr;
  size_t d_temp_size = 0;
  cudaMalloc((void **)&d_in, sizeof(int) * N);
  cudaMalloc((void **)&d_out, sizeof(int) * N);
  cudaMalloc((void **)&d_selected_num, sizeof(int));
  cudaMemcpy((void *)d_in, (void *)data, sizeof(data), cudaMemcpyHostToDevice);
  cub::DeviceSelect::Unique(nullptr, d_temp_size, d_in, d_out, d_selected_num, N);
  cudaMalloc((void **)&d_temp, d_temp_size);
  cub::DeviceSelect::Unique((void *)d_temp, d_temp_size, d_in, d_out, d_selected_num, N);
  cudaDeviceSynchronize();

  int expect_select_num = 5;
  int expect_out[] = {0, 2, 9, 5, 8};

  if (!verify_data(d_selected_num, &expect_select_num, 1)) {
    std::cout << "cub::DeviceSelect::Unique select_num verify failed\n";
    std::cout << "expect:\n";
    print_data<int>(&expect_select_num, 1, true);
    std::cout << "current result:\n";
    print_data<int>(d_selected_num, 1);
    return false;
  }

  if (!verify_data(d_out, (int *)expect_out, expect_select_num)) {
    std::cout << "cub::DeviceSelect::Unique output data verify failed\n";
    std::cout << "expect:\n";
    print_data<int>(expect_out, 1, true);
    std::cout << "current result:\n";
    print_data<int>(d_out, 1);
    return false;
  }
  return true;

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
  bool Result = true;
  Result = test_reduce_1() && Result;
  Result = test_sum_1() && Result;
  Result = test_sum_2() && Result;
  Result = test_min() && Result;
  Result = test_max() && Result;
  Result = test_device_reduce_sum() && Result;
  Result = test_device_scan_inclusive_sum() && Result;
  Result = test_device_scan_inclusive_sum() && Result;
  Result = test_device_select_flagged() && Result;
  Result = test_device_exclusive_scan() && Result;
  Result = test_device_inclusive_scan() && Result;
  Result = test_device_unique() && Result;
  Result = test_device_encode() && Result;
  if(Result) {
    std::cout << "cub_device Pass" << std::endl;
  }
  return 0;
}

