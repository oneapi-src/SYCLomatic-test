// ====------ cub_device_exclusive_sum_by_key.cu --------- *- CUDA -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cub/cub.cuh>
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
  int *d_keys_in = init({0, 0, 1, 1, 1, 2, 2});
  int *d_values_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_values_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected({0, 8, 0, 7, 12, 0, 0});
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSumByKey(d_temp_storage, temp_storage_bytes,
                                     d_keys_in, d_values_in, d_values_out,
                                     num_items);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run exclusive prefix sum
  cub::DeviceScan::ExclusiveSumByKey(d_temp_storage, temp_storage_bytes,
                                     d_keys_in, d_values_in, d_values_out,
                                     num_items);
  std::vector<int> output(num_items, 0);
  cudaMemcpy(output.data(), d_values_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_keys_in);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_temp_storage);
  return std::equal(expected.begin(), expected.end(), output.begin());
}

int main() {
  if (!test()) {
    printf("cub::DeviceScan::ExclusiveSumByKey test failed\n");
    return 1;
  }
  return 0;
}
