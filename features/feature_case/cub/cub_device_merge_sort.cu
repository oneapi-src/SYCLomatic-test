// ====------ cub_device_merge_sort.cu --------------------- *- CUDA -* ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <cub/cub.cuh>
#include <iostream>

int *init(std::initializer_list<int> L) {
  int *Ptr = nullptr;
  cudaMallocManaged(&Ptr, sizeof(int) * L.size());
  cudaMemcpy(Ptr, L.begin(), sizeof(int) * L.size(), cudaMemcpyHostToDevice);
  return Ptr;
}

void dump(const char *s, int *outs, int *expected, int num_items) {
  std::cout << s << ":\t";
  std::ostream_iterator<char> it(std::cout, "\t");
  std::cout << "Output:\t";
  std::copy(outs, outs + num_items, it);
  std::cout << std::endl;
  std::cout << "Expected:\t";
  std::copy(expected, expected + num_items, it);
  std::cout << std::endl;
}

struct CustomOp {
  template <typename DataType>
  __device__ bool operator()(const DataType &lhs, const DataType &rhs) {
    return lhs < rhs;
  }
};

bool sort_pairs() {
  int num_items = 7;
  int *d_keys = init({8, 6, 7, 5, 3, 0, 9});
  int *d_values = init({0, 1, 2, 3, 4, 5, 6});
  int expected_keys[] = {0, 3, 5, 6, 7, 8, 9};
  int expected_values[] = {5, 4, 3, 1, 2, 0, 6};

  CustomOp custom_op;
  void *d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  cub::DeviceMergeSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys,
                                  d_values, num_items, custom_op);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceMergeSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys,
                                  d_values, num_items, custom_op);
  cudaDeviceSynchronize();
  if (std::equal(d_keys, d_keys + num_items, expected_keys) &&
      std::equal(d_values, d_values + num_items, expected_values))
    return true;
  dump("sort_pairs(key)", d_keys, expected_keys, num_items);
  dump("sort_pairs(val)", d_values, expected_values, num_items);
  return false;
}

bool stable_sort_pairs() {
  int num_items = 7;
  int *d_keys = init({8, 6, 7, 5, 3, 0, 9});
  int *d_values = init({0, 1, 2, 3, 4, 5, 6});
  int expected_keys[] = {0, 3, 5, 6, 7, 8, 9};
  int expected_values[] = {5, 4, 3, 1, 2, 0, 6};
  CustomOp custom_op;
  void *d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  cub::DeviceMergeSort::StableSortPairs(d_temp_storage, temp_storage_bytes,
                                        d_keys, d_values, num_items, custom_op);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceMergeSort::StableSortPairs(d_temp_storage, temp_storage_bytes,
                                        d_keys, d_values, num_items, custom_op);
  cudaDeviceSynchronize();
  if (std::equal(d_keys, d_keys + num_items, expected_keys) &&
      std::equal(d_values, d_values + num_items, expected_values))
    return true;
  dump("stable_sort_pairs(key)", d_keys, expected_keys, num_items);
  dump("stable_sort_pairs(val)", d_values, expected_values, num_items);
  return false;
}

bool sort_keys() {
  int num_items = 7;
  int *d_keys = init({8, 6, 7, 5, 3, 0, 9});
  int expected[] = {0, 3, 5, 6, 7, 8, 9};
  CustomOp custom_op;
  void *d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys,
                                 num_items, custom_op);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys,
                                 num_items, custom_op);
  cudaDeviceSynchronize();
  if (std::equal(d_keys, d_keys + num_items, expected))
    return true;
  dump("sort_keys", d_keys, expected, num_items);
  return false;
}

bool stable_sort_keys() {
  int num_items = 7;
  int *d_keys = init({8, 6, 7, 5, 3, 0, 9});
  int expected[] = {0, 3, 5, 6, 7, 8, 9};
  CustomOp custom_op;
  void *d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  cub::DeviceMergeSort::StableSortKeys(d_temp_storage, temp_storage_bytes,
                                       d_keys, num_items, custom_op);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceMergeSort::StableSortKeys(d_temp_storage, temp_storage_bytes,
                                       d_keys, num_items, custom_op);
  cudaDeviceSynchronize();
  if (std::equal(d_keys, d_keys + num_items, expected))
    return true;
  dump("stable_sort_keys", d_keys, expected, num_items);
  return false;
}

bool sort_keys_copy() {
  int num_items = 7;
  int *d_keys = init({8, 6, 7, 5, 3, 0, 9});
  int *d_outs = init({0, 0, 0, 0, 0, 0, 0});
  int expected[] = {0, 3, 5, 6, 7, 8, 9};
  CustomOp custom_op;
  void *d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  cub::DeviceMergeSort::SortKeysCopy(d_temp_storage, temp_storage_bytes, d_keys,
                                     d_outs, num_items, custom_op);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceMergeSort::SortKeysCopy(d_temp_storage, temp_storage_bytes, d_keys,
                                     d_outs, num_items, custom_op);
  cudaDeviceSynchronize();
  if (std::equal(d_outs, d_outs + num_items, expected))
    return true;
  dump("sort_keys_copy", d_outs, expected, num_items);
  return false;
}

int fail = 0;
#define TEST(F)                                                                \
  {                                                                            \
    std::cout << #F;                                                           \
    if (F())                                                                   \
      std::cout << " pass";                                                    \
    else {                                                                     \
      std::cout << " fail";                                                    \
      fail = 1;                                                                \
    }                                                                          \
    std::cout << std::endl;                                                    \
  }

int main() {
  TEST(sort_keys);
  TEST(sort_keys_copy);
  TEST(stable_sort_keys);
  TEST(sort_pairs);
  TEST(stable_sort_pairs);
  assert(not fail);
  return fail;
}
