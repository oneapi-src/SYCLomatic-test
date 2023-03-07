// ====------ cub_device_reduce_max.cu-------------------- *- CUDA -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <cub/cub.cuh>
#include <initializer_list>
#include <cstddef>

template <typename T> T *init(std::initializer_list<T> list) {
  T *p = nullptr;
  cudaMalloc<T>(&p, sizeof(T) * list.size());
  cudaMemcpy(p, list.begin(), sizeof(T) * list.size(), cudaMemcpyHostToDevice);
  return p;
}

int num_items = 7;
int *d_in;
int *d_out;
int out;

bool test1() {
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  cudaFree(d_temp_storage);
  cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  return out == 9;
}

bool test2() {
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cudaStream_t s;
  cudaStreamCreate(&s);
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, s);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, s);
  cudaStreamDestroy(s);
  cudaFree(d_temp_storage);
  cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  return out == 9;
}

int main() {
  d_in = init({8, 6, 7, 5, -3, 0, 9});
  d_out = init({0});
  bool res = test1();
  res =  test2() && res;
  cudaFree(d_in);
  cudaFree(d_out);

  if (!res) {
    printf("cub::DeviceReduce::Max test failed!\n");
    return 1;
  }
  return 0;
}
