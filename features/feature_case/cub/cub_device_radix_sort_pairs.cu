// ====------ cub_device_radix_sort_pairs.cu ------------- *- CUDA -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cub/cub.cuh>
#include <initializer_list>
#include <stdio.h>
#include <vector>

template <typename T> T *init(std::initializer_list<T> list) {
  T *arr = nullptr;
  cudaMalloc(&arr, sizeof(T) * list.size());
  cudaMemcpy(arr, list.begin(), sizeof(T) * list.size(),
             cudaMemcpyHostToDevice);
  return arr;
}

bool test() {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  int *d_values_in = init({0, 1, 2, 3, 4, 5, 6});
  int *d_values_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{0, 3, 5, 6, 7, 8, 9};
  std::vector<int> expected_values_out{5, 4, 3, 1, 2, 0, 6};

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in,
                                  d_keys_out, d_values_in, d_values_out,
                                  num_items);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in,
                                  d_keys_out, d_values_in, d_values_out,
                                  num_items);
  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(values_out.data(), d_values_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_temp_storage);
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin()) &&
         std::equal(expected_values_out.begin(), expected_values_out.end(),
                    values_out.begin());
}

bool test1() {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  int *d_values_in = init({0, 1, 2, 3, 4, 5, 6});
  int *d_values_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{3, 0, 6, 7, 5, 8, 9};
  std::vector<int> expected_values_out{4, 5, 1, 2, 3, 0, };

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in,
                                  d_keys_out, d_values_in, d_values_out,
                                  num_items, 2);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in,
                                  d_keys_out, d_values_in, d_values_out,
                                  num_items, 2);
  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(values_out.data(), d_values_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_temp_storage);
  
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin()) &&
         std::equal(expected_values_out.begin(), expected_values_out.end(),
                    values_out.begin());
}

bool test2() {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  int *d_values_in = init({0, 1, 2, 3, 4, 5, 6});
  int *d_values_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{3, 0, 6, 7, 5, 8, 9};
  std::vector<int> expected_values_out{4, 5, 1, 2, 3, 0, 6};

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in,
                                  d_keys_out, d_values_in, d_values_out,
                                  num_items, 2, 4);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in,
                                  d_keys_out, d_values_in, d_values_out,
                                  num_items, 2, 4);
  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(values_out.data(), d_values_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_temp_storage);
  
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin()) &&
         std::equal(expected_values_out.begin(), expected_values_out.end(),
                    values_out.begin());
}

bool test3() {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  int *d_values_in = init({0, 1, 2, 3, 4, 5, 6});
  int *d_values_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{3, 0, 6, 7, 5, 8, 9};
  std::vector<int> expected_values_out{4, 5, 1, 2, 3, 0, 6};

  cudaStream_t s = nullptr;
  cudaStreamCreate(&s);
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in,
                                  d_keys_out, d_values_in, d_values_out,
                                  num_items, 2, 4, s);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in,
                                  d_keys_out, d_values_in, d_values_out,
                                  num_items, 2, 4, s);
  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(values_out.data(), d_values_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_temp_storage);
  cudaStreamDestroy(s);
  
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin()) &&
         std::equal(expected_values_out.begin(), expected_values_out.end(),
                    values_out.begin());
}

bool test_descending() {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  int *d_values_in = init({0, 1, 2, 3, 4, 5, 6});
  int *d_values_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{9, 8, 7, 6, 5, 3, 0};
  std::vector<int> expected_values_out{6, 0, 2, 1, 3, 4, 5};

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
                                            d_keys_in, d_keys_out, d_values_in,
                                            d_values_out, num_items);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
                                            d_keys_in, d_keys_out, d_values_in,
                                            d_values_out, num_items);
  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(values_out.data(), d_values_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_temp_storage);
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin()) &&
         std::equal(expected_values_out.begin(), expected_values_out.end(),
                    values_out.begin());
}

bool test_descending1() {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  int *d_values_in = init({0, 1, 2, 3, 4, 5, 6});
  int *d_values_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{8, 9, 6, 7, 5, 3, 0};
  std::vector<int> expected_values_out{0, 6, 1, 2, 3, 4, 5};

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
                                            d_keys_in, d_keys_out, d_values_in,
                                            d_values_out, num_items, 2);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
                                            d_keys_in, d_keys_out, d_values_in,
                                            d_values_out, num_items, 2);
  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(values_out.data(), d_values_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_temp_storage);
  
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin()) &&
         std::equal(expected_values_out.begin(), expected_values_out.end(),
                    values_out.begin());
}

bool test_descending2() {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  int *d_values_in = init({0, 1, 2, 3, 4, 5, 6});
  int *d_values_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{8, 9, 6, 7, 5, 3, 0};
  std::vector<int> expected_values_out{0, 6, 1, 2, 3, 4, 5};

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
                                            d_keys_in, d_keys_out, d_values_in,
                                            d_values_out, num_items, 2, 4);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
                                            d_keys_in, d_keys_out, d_values_in,
                                            d_values_out, num_items, 2, 4);
  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(values_out.data(), d_values_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_temp_storage);
  
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin()) &&
         std::equal(expected_values_out.begin(), expected_values_out.end(),
                    values_out.begin());
}

bool test_descending3() {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  int *d_values_in = init({0, 1, 2, 3, 4, 5, 6});
  int *d_values_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{8, 9, 6, 7, 5, 3, 0};
  std::vector<int> expected_values_out{0, 6, 1, 2, 3, 4, 5};
  cudaStream_t s = nullptr;
  cudaStreamCreate(&s);
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
                                            d_keys_in, d_keys_out, d_values_in,
                                            d_values_out, num_items, 2, 4, s);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
                                            d_keys_in, d_keys_out, d_values_in,
                                            d_values_out, num_items, 2, 4, s);
  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(values_out.data(), d_values_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_temp_storage);
  cudaStreamDestroy(s);
  
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin()) &&
         std::equal(expected_values_out.begin(), expected_values_out.end(),
                    values_out.begin());
}

int main() {
  int res = 0;
  if (!test()) {
    printf("cub::DeviceRadixSort::SortPairs failed\n");
    res = 1;
  }
  if (!test1()) {
    printf("cub::DeviceRadixSort::SortPairs failed\n");
    res = 1;
  }
  if (!test2()) {
    printf("cub::DeviceRadixSort::SortPairs failed\n");
    res = 1;
  }
  if (!test3()) {
    printf("cub::DeviceRadixSort::SortPairs failed\n");
    res = 1;
  }

  if (!test_descending()) {
    printf("cub::DeviceRadixSort::SortPairsDescending failed\n");
    res = 1;
  }

  if (!test_descending1()) {
    printf("cub::DeviceRadixSort::SortPairsDescending failed\n");
    res = 1;
  }

  if (!test_descending2()) {
    printf("cub::DeviceRadixSort::SortPairsDescending failed\n");
    res = 1;
  }

  if (!test_descending3()) {
    printf("cub::DeviceRadixSort::SortPairsDescending failed\n");
    res = 1;
  }

  return res;
}