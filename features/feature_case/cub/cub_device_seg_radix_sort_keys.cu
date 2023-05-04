#include <cstdlib>
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
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{6, 7, 8, 0, 3, 5, 9};
  // clang-format on

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
      num_segments, d_offsets, d_offsets + 1);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
      num_segments, d_offsets, d_offsets + 1);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin());
}

bool test1() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{6, 7, 8, 3, 0, 5, 9};
  // clang-format on

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
      num_segments, d_offsets, d_offsets + 1, 2);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
      num_segments, d_offsets, d_offsets + 1, 2);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);

  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin());
}

bool test2() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{6, 7, 8, 3, 0, 5, 9};
  // clang-format on

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
      num_segments, d_offsets, d_offsets + 1, 2, 4);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
      num_segments, d_offsets, d_offsets + 1, 2, 4);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);

  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);

  cudaFree(d_temp_storage);

  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin());
}

bool test3() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{6, 7, 8, 3, 0, 5, 9};
  // clang-format on
  cudaStream_t s;
  cudaStreamCreate(&s);
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
      num_segments, d_offsets, d_offsets + 1, 2, 4, s);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
      num_segments, d_offsets, d_offsets + 1, 2, 4, s);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  cudaStreamDestroy(s);

  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin());
}

bool testDescennding() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{8, 7, 6, 9, 5, 3, 0};
  // clang-format on

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
      num_segments, d_offsets, d_offsets + 1);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
      num_segments, d_offsets, d_offsets + 1);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin());
}

bool testDescennding1() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{8, 6, 7, 9, 5, 3, 0};
  // clang-format on

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
      num_segments, d_offsets, d_offsets + 1, 2);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
      num_segments, d_offsets, d_offsets + 1, 2);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);

  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin());
}

bool testDescennding2() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{8, 6, 7, 9, 5, 3, 0};
  // clang-format on

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
      num_segments, d_offsets, d_offsets + 1, 2, 4);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
      num_segments, d_offsets, d_offsets + 1, 2, 4);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);

  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin());
}

bool testDescennding3() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{8, 6, 7, 9, 5, 3, 0};
  // clang-format on
  cudaStream_t s;
  cudaStreamCreate(&s);
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
      num_segments, d_offsets, d_offsets + 1, 2, 4, s);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
      num_segments, d_offsets, d_offsets + 1, 2, 4, s);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  cudaStreamDestroy(s);

  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin());
}

bool testDoubleBuffer() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  cub::DoubleBuffer<int> d_keys(d_keys_in, d_keys_out);
  std::vector<int> expected_keys_out{6, 7, 8, 0, 3, 5, 9};
  // clang-format on

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                          d_keys, num_items, num_segments,
                                          d_offsets, d_offsets + 1);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                          d_keys, num_items, num_segments,
                                          d_offsets, d_offsets + 1);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys.Current(), sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin());
}

bool testDoubleBuffer1() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  cub::DoubleBuffer<int> d_keys(d_keys_in, d_keys_out);
  std::vector<int> expected_keys_out{6, 7, 8, 3, 0, 5, 9};
  // clang-format on

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                          d_keys, num_items, num_segments,
                                          d_offsets, d_offsets + 1, 2);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                          d_keys, num_items, num_segments,
                                          d_offsets, d_offsets + 1, 2);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys.Current(), sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);

  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin());
}

bool testDoubleBuffer2() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  cub::DoubleBuffer<int> d_keys(d_keys_in, d_keys_out);
  std::vector<int> expected_keys_out{6, 7, 8, 3, 0, 5, 9};
  // clang-format on

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                          d_keys, num_items, num_segments,
                                          d_offsets, d_offsets + 1, 2, 4);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                          d_keys, num_items, num_segments,
                                          d_offsets, d_offsets + 1, 2, 4);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys.Current(), sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);

  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);

  cudaFree(d_temp_storage);

  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin());
}

bool testDoubleBuffer3() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  cub::DoubleBuffer<int> d_keys(d_keys_in, d_keys_out);
  std::vector<int> expected_keys_out{6, 7, 8, 3, 0, 5, 9};
  // clang-format on
  cudaStream_t s;
  cudaStreamCreate(&s);
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                          d_keys, num_items, num_segments,
                                          d_offsets, d_offsets + 1, 2, 4, s);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                          d_keys, num_items, num_segments,
                                          d_offsets, d_offsets + 1, 2, 4, s);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys.Current(), sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  cudaStreamDestroy(s);

  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin());
}

bool testDoubleBufferDescennding() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  cub::DoubleBuffer<int> d_keys(d_keys_in, d_keys_out);
  std::vector<int> expected_keys_out{8, 7, 6, 9, 5, 3, 0};
  // clang-format on

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments,
      d_offsets, d_offsets + 1);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments,
      d_offsets, d_offsets + 1);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys.Current(), sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin());
}

bool testDoubleBufferDescennding1() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  cub::DoubleBuffer<int> d_keys(d_keys_in, d_keys_out);
  std::vector<int> expected_keys_out{8, 6, 7, 9, 5, 3, 0};
  // clang-format on

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments,
      d_offsets, d_offsets + 1, 2);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments,
      d_offsets, d_offsets + 1, 2);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys.Current(), sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);

  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin());
}

bool testDoubleBufferDescennding2() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  cub::DoubleBuffer<int> d_keys(d_keys_in, d_keys_out);
  std::vector<int> expected_keys_out{8, 6, 7, 9, 5, 3, 0};
  // clang-format on

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments,
      d_offsets, d_offsets + 1, 2, 4);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments,
      d_offsets, d_offsets + 1, 2, 4);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys.Current(), sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);

  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin());
}

bool testDoubleBufferDescennding3() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  cub::DoubleBuffer<int> d_keys(d_keys_in, d_keys_out);
  std::vector<int> expected_keys_out{8, 6, 7, 9, 5, 3, 0};
  // clang-format on
  cudaStream_t s;
  cudaStreamCreate(&s);
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments,
      d_offsets, d_offsets + 1, 2, 4, s);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortKeysDescending(
      d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments,
      d_offsets, d_offsets + 1, 2, 4, s);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys.Current(), sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  cudaStreamDestroy(s);

  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin());
}

int main() {
  if (!test()) {
    printf("cub::DeviceSegmentedRadixSort::SortKeys test failed\n");
    return EXIT_FAILURE;
  }
  if (!test1()) {
    printf("cub::DeviceSegmentedRadixSort::SortKeys test1 failed\n");
    return EXIT_FAILURE;
  }
  if (!test2()) {
    printf("cub::DeviceSegmentedRadixSort::SortKeys test2 failed\n");
    return EXIT_FAILURE;
  }
  if (!test3()) {
    printf("cub::DeviceSegmentedRadixSort::SortKeys test3 failed\n");
    return EXIT_FAILURE;
  }

  if (!testDescennding()) {
    printf("cub::DeviceSegmentedRadixSort::SortKeysDescending testDescennding "
           "failed\n");
    return EXIT_FAILURE;
  }

  if (!testDescennding1()) {
    printf("cub::DeviceSegmentedRadixSort::SortKeysDescending testDescennding1 "
           "failed\n");
    return EXIT_FAILURE;
  }

  if (!testDescennding2()) {
    printf("cub::DeviceSegmentedRadixSort::SortKeysDescending testDescennding2 "
           "failed\n");
    return EXIT_FAILURE;
  }

  if (!testDescennding3()) {
    printf("cub::DeviceSegmentedRadixSort::SortKeysDescending testDescennding3 "
           "failed\n");
    return EXIT_FAILURE;
  }

  if (!testDoubleBuffer()) {
    printf("cub::DeviceSegmentedRadixSort::SortKeys testDoubleBuffer failed\n");
    return EXIT_FAILURE;
  }
  if (!testDoubleBuffer1()) {
    printf(
        "cub::DeviceSegmentedRadixSort::SortKeys testDoubleBuffer1 failed\n");
    return EXIT_FAILURE;
  }
  if (!testDoubleBuffer2()) {
    printf(
        "cub::DeviceSegmentedRadixSort::SortKeys testDoubleBuffer2 failed\n");
    return EXIT_FAILURE;
  }
  if (!testDoubleBuffer3()) {
    printf(
        "cub::DeviceSegmentedRadixSort::SortKeys testDoubleBuffer3 failed\n");
    return EXIT_FAILURE;
  }

  if (!testDoubleBufferDescennding()) {
    printf("cub::DeviceSegmentedRadixSort::SortKeysDescending "
           "testDoubleBufferDescennding failed\n");
    return EXIT_FAILURE;
  }

  if (!testDoubleBufferDescennding1()) {
    printf("cub::DeviceSegmentedRadixSort::SortKeysDescending "
           "testDoubleBufferDescennding1 failed\n");
    return EXIT_FAILURE;
  }

  if (!testDoubleBufferDescennding2()) {
    printf("cub::DeviceSegmentedRadixSort::SortKeysDescending "
           "testDoubleBufferDescennding2 failed\n");
    return EXIT_FAILURE;
  }

  if (!testDoubleBufferDescennding3()) {
    printf("cub::DeviceSegmentedRadixSort::SortKeysDescending "
           "testDoubleBufferDescennding3 failed\n");
    return EXIT_FAILURE;
  }

  return 0;
}
