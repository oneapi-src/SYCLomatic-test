#include <cub/cub.cuh>
#include <initializer_list>
#include <stdio.h>
#include <vector>
#include <iostream>

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
  std::vector<int> expected_keys_out{0, 3, 5, 6, 7, 8, 9};

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,
                                 d_keys_out, num_items);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,
                                 d_keys_out, num_items);
  std::vector<int> keys_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin());
}

bool test1() {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{3, 0, 6, 7, 5, 8, 9};

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,
                                 d_keys_out, num_items, 2);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,
                                 d_keys_out, num_items, 2);
  std::vector<int> keys_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin());
}

bool test2() {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{3, 0, 6, 7, 5, 8, 9};

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,
                                 d_keys_out, num_items, 2, 4);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,
                                 d_keys_out, num_items, 2, 4);
  std::vector<int> keys_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin());
}

bool test3() {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{3, 0, 6, 7, 5, 8, 9};
  cudaStream_t s = nullptr;
  cudaStreamCreate(&s);

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,
                                 d_keys_out, num_items, 2, 4, s);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,
                                 d_keys_out, num_items, 2, 4, s);
  std::vector<int> keys_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  cudaStreamDestroy(s);
  
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin());
}

bool test_descending() {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{9, 8, 7, 6, 5, 3, 0};

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes,
                                           d_keys_in, d_keys_out, num_items);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes,
                                           d_keys_in, d_keys_out, num_items);
  std::vector<int> keys_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin());
}

bool test_descending1() {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{8, 9, 6, 7, 5, 3, 0};

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes,
                                           d_keys_in, d_keys_out, num_items, 2);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes,
                                           d_keys_in, d_keys_out, num_items, 2);
  std::vector<int> keys_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin());
}

bool test_descending2() {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{8, 9, 6, 7, 5, 3, 0};

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes,
                                           d_keys_in, d_keys_out, num_items, 2, 4);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes,
                                           d_keys_in, d_keys_out, num_items, 2, 4);
  std::vector<int> keys_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin());
}

bool test_descending3() {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{8, 9, 6, 7, 5, 3, 0};
  cudaStream_t s = nullptr;
  cudaStreamCreate(&s);

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes,
                                           d_keys_in, d_keys_out, num_items, 2, 4, s);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes,
                                           d_keys_in, d_keys_out, num_items, 2, 4, s);
  std::vector<int> keys_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  cudaStreamDestroy(s);
  
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin());
}

int main() {
  int res = 0;
  if (!test()) {
    printf("cub::DeviceRadixSort::SortKeys failed\n");
    res = 1;
  }
  if (!test1()) {
    printf("cub::DeviceRadixSort::SortKeys failed\n");
    res = 1;
  }
  if (!test2()) {
    printf("cub::DeviceRadixSort::SortKeys failed\n");
    res = 1;
  }
  if (!test3()) {
    printf("cub::DeviceRadixSort::SortKeys failed\n");
    res = 1;
  }

  if (!test_descending()) {
    printf("cub::DeviceRadixSort::SortKeysDescending failed\n");
    res = 1;
  }

  if (!test_descending1()) {
    printf("cub::DeviceRadixSort::SortKeysDescending failed\n");
    res = 1;
  }

  if (!test_descending2()) {
    printf("cub::DeviceRadixSort::SortKeysDescending failed\n");
    res = 1;
  }

  if (!test_descending3()) {
    printf("cub::DeviceRadixSort::SortKeysDescending failed\n");
    res = 1;
  }

  return res;
}