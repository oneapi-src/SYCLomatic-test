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
  int  *d_values_in       = init({0, 1, 2, 3, 4, 5, 6});
  int  *d_values_out      = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{6, 7, 8, 0, 3, 5, 9};
  std::vector<int> expected_values_out{1, 2, 0, 5, 4, 3, 6};
  // clang-format on

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(values_out.data(), d_values_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_temp_storage);
  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin()) &&
         std::equal(values_out.begin(), values_out.end(),
                    expected_values_out.begin());
}

bool test1() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  int  *d_values_in       = init({0, 1, 2, 3, 4, 5, 6});
  int  *d_values_out      = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{6, 7, 8, 3, 0, 5, 9};
  std::vector<int> expected_values_out{1, 2, 0, 4, 5, 3, 6};
  // clang-format on

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 2);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 2);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(values_out.data(), d_values_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_temp_storage);
  
  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin()) &&
         std::equal(values_out.begin(), values_out.end(),
                    expected_values_out.begin());
}

bool test2() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  int  *d_values_in       = init({0, 1, 2, 3, 4, 5, 6});
  int  *d_values_out      = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{6, 7, 8, 3, 0, 5, 9};
  std::vector<int> expected_values_out{1, 2, 0, 4, 5, 3, 6};
  // clang-format on

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 2, 4);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 2, 4);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(values_out.data(), d_values_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_temp_storage);
  
  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin()) &&
         std::equal(values_out.begin(), values_out.end(),
                    expected_values_out.begin());
}

bool test3() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  int  *d_values_in       = init({0, 1, 2, 3, 4, 5, 6});
  int  *d_values_out      = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{6, 7, 8, 3, 0, 5, 9};
  std::vector<int> expected_values_out{1, 2, 0, 4, 5, 3, 6};
  // clang-format on
  cudaStream_t s;
  cudaStreamCreate(&s);
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 2, 4, s);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 2, 4, s);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(values_out.data(), d_values_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_temp_storage);
  cudaStreamDestroy(s);
  
  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin()) &&
         std::equal(values_out.begin(), values_out.end(),
                    expected_values_out.begin());
}

bool test_descending() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  int  *d_values_in       = init({0, 1, 2, 3, 4, 5, 6});
  int  *d_values_out      = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{8, 7, 6, 9, 5, 3, 0};
  std::vector<int> expected_values_out{0, 2, 1, 6, 3, 4, 5};
  // clang-format on

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(values_out.data(), d_values_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_temp_storage);
  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin()) &&
         std::equal(values_out.begin(), values_out.end(),
                    expected_values_out.begin());
}

bool test_descending1() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  int  *d_values_in       = init({0, 1, 2, 3, 4, 5, 6});
  int  *d_values_out      = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{8, 6, 7, 9, 5, 3, 0};
  std::vector<int> expected_values_out{0, 1, 2, 6, 3, 4, 5};
  // clang-format on

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 2);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 2);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(values_out.data(), d_values_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_temp_storage);
  
  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin()) &&
         std::equal(values_out.begin(), values_out.end(),
                    expected_values_out.begin());
}

bool test_descending2() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  int  *d_values_in       = init({0, 1, 2, 3, 4, 5, 6});
  int  *d_values_out      = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{8, 6, 7, 9, 5, 3, 0};
  std::vector<int> expected_values_out{0, 1, 2, 6, 3, 4, 5};
  // clang-format on

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 2, 4);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 2, 4);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(values_out.data(), d_values_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_temp_storage);
  
  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin()) &&
         std::equal(values_out.begin(), values_out.end(),
                    expected_values_out.begin());
}

bool test_descending3() {
  // clang-format off
  // Declare, allocate, and initialize device-accessible pointers for sorting data
  int  num_items          = 7;
  int  num_segments       = 3;
  int  *d_offsets         = init({0, 3, 3, 7});
  int  *d_keys_in         = init({8, 6, 7, 5, 3, 0, 9});
  int  *d_keys_out        = init({0, 0, 0, 0, 0, 0, 0});
  int  *d_values_in       = init({0, 1, 2, 3, 4, 5, 6});
  int  *d_values_out      = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{8, 6, 7, 9, 5, 3, 0};
  std::vector<int> expected_values_out{0, 1, 2, 6, 3, 4, 5};
  // clang-format on
  cudaStream_t s;
  cudaStreamCreate(&s);
  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 2, 4, s);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 2, 4, s);

  std::vector<int> keys_out(num_items), values_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(values_out.data(), d_values_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  cudaFree(d_offsets);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_temp_storage);
  cudaStreamDestroy(s);
  
  return std::equal(keys_out.begin(), keys_out.end(),
                    expected_keys_out.begin()) &&
         std::equal(values_out.begin(), values_out.end(),
                    expected_values_out.begin());
}

int main() {
  int res = 0;
  if (!test()) {
    printf("cub::DeviceSegmentedRadixSort::SortPairs failed\n");
    res = 1;
  }
  if (!test1()) {
    printf("cub::DeviceSegmentedRadixSort::SortPairs failed\n");
    res = 1;
  }
  if (!test2()) {
    printf("cub::DeviceSegmentedRadixSort::SortPairs failed\n");
    res = 1;
  }
  if (!test3()) {
    printf("cub::DeviceSegmentedRadixSort::SortPairs failed\n");
    res = 1;
  }

  if (!test_descending()) {
    printf("cub::DeviceSegmentedRadixSort::SortPairsDescending failed\n");
    res = 1;
  }

  if (!test_descending1()) {
    printf("cub::DeviceSegmentedRadixSort::SortPairsDescending failed\n");
    res = 1;
  }

  if (!test_descending2()) {
    printf("cub::DeviceSegmentedRadixSort::SortPairsDescending failed\n");
    res = 1;
  }

  if (!test_descending3()) {
    printf("cub::DeviceSegmentedRadixSort::SortPairsDescending failed\n");
    res = 1;
  }

  return res;
}
