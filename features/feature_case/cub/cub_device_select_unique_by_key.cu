// ====------ cub_device_select_unqiue_by_key.cu---------- *- CUDA -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cub/cub.cuh>
#include <vector>
#include <iostream>

template <typename T> T *init(std::initializer_list<T> list) {
  T *p = nullptr;
  cudaMalloc(&p, sizeof(T) * list.size());
  cudaMemcpy(p, list.begin(), sizeof(T) * list.size(), cudaMemcpyHostToDevice);
  return p;
}

bool test1() {

  int num_items = 8;                                    // e.g., 8
  int *d_keys_in = init({0, 2, 2, 9, 5, 5, 5, 8});      // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
  int *d_values_in = init({1, 2, 3, 4, 5, 6, 7, 8});    // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0, 0});     // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  int *d_values_out = init({0, 0, 0, 0, 0, 0, 0, 0});   // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  int *d_num_selected_out = init({0});                  // e.g., [ ]
  
  int expected_num_sel = 5;
  std::vector<int> expected_keys{0, 2, 9, 5, 8};
  std::vector<int> expected_vals{1, 2, 4, 5, 8};

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::UniqueByKey(d_temp_storage, temp_storage_bytes, d_keys_in,
                                 d_values_in, d_keys_out, d_values_out,
                                 d_num_selected_out, num_items);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run selection
  cub::DeviceSelect::UniqueByKey(d_temp_storage, temp_storage_bytes, d_keys_in,
                                 d_values_in, d_keys_out, d_values_out,
                                 d_num_selected_out, num_items);

  int num_sel = 0;
  cudaMemcpy(&num_sel, d_num_selected_out, sizeof(int), cudaMemcpyDeviceToHost);

  if (num_sel != expected_num_sel) {
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_values_in);
    cudaFree(d_values_out);
    cudaFree(d_num_selected_out);
    return false;
  }

  std::vector<int> keys(num_sel), vals(num_sel);

  cudaMemcpy(keys.data(), d_keys_out, sizeof(int) * num_sel, cudaMemcpyDeviceToHost);
  cudaMemcpy(vals.data(), d_values_out, sizeof(int) * num_sel, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_num_selected_out);
  return std::equal(keys.begin(), keys.end(), expected_keys.begin()) &&
         std::equal(vals.begin(), vals.end(), expected_vals.begin());
}

bool test2() {

  int num_items = 8;                                    // e.g., 8
  int *d_keys_in = init({0, 2, 2, 9, 5, 5, 5, 8});      // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
  int *d_values_in = init({1, 2, 3, 4, 5, 6, 7, 8});    // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0, 0});     // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  int *d_values_out = init({0, 0, 0, 0, 0, 0, 0, 0});   // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  int *d_num_selected_out = init({0});                  // e.g., [ ]

  int expected_num_sel = 5;
  std::vector<int> expected_keys{0, 2, 9, 5, 8};
  std::vector<int> expected_vals{1, 2, 4, 5, 8};

  cudaStream_t s;
  cudaStreamCreate(&s);

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::UniqueByKey(d_temp_storage, temp_storage_bytes, d_keys_in,
                                 d_values_in, d_keys_out, d_values_out,
                                 d_num_selected_out, num_items, s);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run selection
  cub::DeviceSelect::UniqueByKey(d_temp_storage, temp_storage_bytes, d_keys_in,
                                 d_values_in, d_keys_out, d_values_out,
                                 d_num_selected_out, num_items, s);
  cudaFree(d_temp_storage);

  int num_sel = 0;
  cudaMemcpy(&num_sel, d_num_selected_out, sizeof(int), cudaMemcpyDeviceToHost);

  if (num_sel != expected_num_sel) {
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_values_in);
    cudaFree(d_values_out);
    cudaFree(d_num_selected_out);
    cudaStreamDestroy(s);
    return false;
  }

  std::vector<int> keys(num_sel), vals(num_sel);

  cudaMemcpy(keys.data(), d_keys_out, sizeof(int) * num_sel, cudaMemcpyDeviceToHost);
  cudaMemcpy(vals.data(), d_values_out, sizeof(int) * num_sel, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_num_selected_out);
  cudaStreamDestroy(s);
  return std::equal(keys.begin(), keys.begin(), expected_keys.begin()) &&
         std::equal(vals.begin(), vals.end(), expected_vals.begin());
}

int main() {
  bool res = test1();
  res = test2() && res;
  if (!res) {
    printf("cub::DeviceSelect::UniqueByKey failed\n");
    return 1;
  }
  return 0;
}
