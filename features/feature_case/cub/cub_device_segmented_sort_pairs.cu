#include <cstdlib>
#include <iostream>
#include <vector>

#include <cub/cub.cuh>
#include <cub/device/device_segmented_sort.cuh>

bool testStableSortPairs() {
  // Declare, allocate, and initialize device-accessible pointers
  // for sorting data
  int num_items;     // e.g., 7
  int num_segments;  // e.g., 3
  int *d_offsets;    // e.g., [0, 3, 3, 7]
  int *d_keys_in;    // e.g., [8, 6, 7, 5, 3, 0, 9]
  int *d_keys_out;   // e.g., [-, -, -, -, -, -, -]
  int *d_values_in;  // e.g., [0, 1, 2, 3, 4, 5, 6]
  int *d_values_out; // e.g., [-, -, -, -, -, -, -]

  num_items = 7;
  num_segments = 3;
  cudaMallocManaged(&d_offsets, (num_segments + 1) * sizeof(*d_offsets));
  cudaMallocManaged(&d_keys_in, num_items * sizeof(*d_keys_in));
  cudaMallocManaged(&d_keys_out, num_items * sizeof(*d_keys_out));
  cudaMallocManaged(&d_values_in, num_items * sizeof(*d_values_in));
  cudaMallocManaged(&d_values_out, num_items * sizeof(*d_values_out));

  d_offsets[0] = 0;
  d_offsets[1] = 3;
  d_offsets[2] = 3;
  d_offsets[3] = 7;

  d_keys_in[0] = 8;
  d_keys_in[1] = 6;
  d_keys_in[2] = 7;
  d_keys_in[3] = 5;
  d_keys_in[4] = 3;
  d_keys_in[5] = 0;
  d_keys_in[6] = 9;

  d_values_in[0] = 0;
  d_values_in[1] = 1;
  d_values_in[2] = 2;
  d_values_in[3] = 3;
  d_values_in[4] = 4;
  d_values_in[5] = 5;
  d_values_in[6] = 6;

  cudaDeviceSynchronize();

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedSort::StableSortPairs(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);

  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Run sorting operation
  cub::DeviceSegmentedSort::StableSortPairs(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);

  // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
  // d_values_out          <-- [1, 2, 0, 5, 4, 3, 6]

  std::vector<int> d_keys_out_expected = {6, 7, 8, 0, 3, 5, 9};
  std::vector<int> d_values_out_expected = {1, 2, 0, 5, 4, 3, 6};

  cudaDeviceSynchronize();
  for (int i = 0; i < num_items; i++) {
    if (d_keys_out[i] != d_keys_out_expected[i] ||
        d_values_out[i] != d_values_out_expected[i]) {
      return false;
    }
  }

  return true;
}

bool testStableSortPairsDB() {
  // Declare, allocate, and initialize device-accessible pointers
  // for sorting data
  int num_items;     // e.g., 7
  int num_segments;  // e.g., 3
  int *d_offsets;    // e.g., [0, 3, 3, 7]
  int *d_keys_in;    // e.g., [8, 6, 7, 5, 3, 0, 9]
  int *d_keys_out;   // e.g., [-, -, -, -, -, -, -]
  int *d_values_in;  // e.g., [0, 1, 2, 3, 4, 5, 6]
  int *d_values_out; // e.g., [-, -, -, -, -, -, -]

  num_items = 7;
  num_segments = 3;
  cudaMallocManaged(&d_offsets, (num_segments + 1) * sizeof(*d_offsets));
  cudaMallocManaged(&d_keys_in, num_items * sizeof(*d_keys_in));
  cudaMallocManaged(&d_keys_out, num_items * sizeof(*d_keys_out));
  cudaMallocManaged(&d_values_in, num_items * sizeof(*d_values_in));
  cudaMallocManaged(&d_values_out, num_items * sizeof(*d_values_out));

  d_offsets[0] = 0;
  d_offsets[1] = 3;
  d_offsets[2] = 3;
  d_offsets[3] = 7;

  d_keys_in[0] = 8;
  d_keys_in[1] = 6;
  d_keys_in[2] = 7;
  d_keys_in[3] = 5;
  d_keys_in[4] = 3;
  d_keys_in[5] = 0;
  d_keys_in[6] = 9;

  d_values_in[0] = 0;
  d_values_in[1] = 1;
  d_values_in[2] = 2;
  d_values_in[3] = 3;
  d_values_in[4] = 4;
  d_values_in[5] = 5;
  d_values_in[6] = 6;

  cub::DoubleBuffer<int> d_keys(d_keys_in, d_keys_out);
  cub::DoubleBuffer<int> d_values(d_values_in, d_values_out);

  cudaDeviceSynchronize();

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedSort::StableSortPairs(
      d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items,
      num_segments, d_offsets, d_offsets + 1);

  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Run sorting operation
  cub::DeviceSegmentedSort::StableSortPairs(
      d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items,
      num_segments, d_offsets, d_offsets + 1);

  // d_keys.Current()      <-- [6, 7, 8, 0, 3, 5, 9]
  // d_values.Current()    <-- [1, 2, 0, 5, 4, 3, 6]

  std::vector<int> d_keys_out_expected = {6, 7, 8, 0, 3, 5, 9};
  std::vector<int> d_values_out_expected = {1, 2, 0, 5, 4, 3, 6};

  cudaDeviceSynchronize();
  for (int i = 0; i < num_items; i++) {
    if (d_keys.Current()[i] != d_keys_out_expected[i] ||
        d_values.Current()[i] != d_values_out_expected[i]) {
      return false;
    }
  }

  return true;
}

bool testStableSortPairsDescending() {
  // Declare, allocate, and initialize device-accessible pointers
  // for sorting data
  int num_items;     // e.g., 7
  int num_segments;  // e.g., 3
  int *d_offsets;    // e.g., [0, 3, 3, 7]
  int *d_keys_in;    // e.g., [8, 6, 7, 5, 3, 0, 9]
  int *d_keys_out;   // e.g., [-, -, -, -, -, -, -]
  int *d_values_in;  // e.g., [0, 1, 2, 3, 4, 5, 6]
  int *d_values_out; // e.g., [-, -, -, -, -, -, -]

  num_items = 7;
  num_segments = 3;
  cudaMallocManaged(&d_offsets, (num_segments + 1) * sizeof(*d_offsets));
  cudaMallocManaged(&d_keys_in, num_items * sizeof(*d_keys_in));
  cudaMallocManaged(&d_keys_out, num_items * sizeof(*d_keys_out));
  cudaMallocManaged(&d_values_in, num_items * sizeof(*d_values_in));
  cudaMallocManaged(&d_values_out, num_items * sizeof(*d_values_out));

  d_offsets[0] = 0;
  d_offsets[1] = 3;
  d_offsets[2] = 3;
  d_offsets[3] = 7;

  d_keys_in[0] = 8;
  d_keys_in[1] = 6;
  d_keys_in[2] = 7;
  d_keys_in[3] = 5;
  d_keys_in[4] = 3;
  d_keys_in[5] = 0;
  d_keys_in[6] = 9;

  d_values_in[0] = 0;
  d_values_in[1] = 1;
  d_values_in[2] = 2;
  d_values_in[3] = 3;
  d_values_in[4] = 4;
  d_values_in[5] = 5;
  d_values_in[6] = 6;

  cudaDeviceSynchronize();

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedSort::StableSortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);

  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Run sorting operation
  cub::DeviceSegmentedSort::StableSortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);

  // d_keys_out            <-- [8, 7, 6, 9, 5, 3, 0]
  // d_values_out          <-- [0, 2, 1, 6, 3, 4, 5]

  std::vector<int> d_keys_out_expected = {8, 7, 6, 9, 5, 3, 0};
  std::vector<int> d_values_out_expected = {0, 2, 1, 6, 3, 4, 5};

  cudaDeviceSynchronize();
  for (int i = 0; i < num_items; i++) {
    if (d_keys_out[i] != d_keys_out_expected[i] ||
        d_values_out[i] != d_values_out_expected[i]) {
      return false;
    }
  }

  return true;
}

bool testStableSortPairsDescendingDB() {
  // Declare, allocate, and initialize device-accessible pointers
  // for sorting data
  int num_items;     // e.g., 7
  int num_segments;  // e.g., 3
  int *d_offsets;    // e.g., [0, 3, 3, 7]
  int *d_keys_in;    // e.g., [8, 6, 7, 5, 3, 0, 9]
  int *d_keys_out;   // e.g., [-, -, -, -, -, -, -]
  int *d_values_in;  // e.g., [0, 1, 2, 3, 4, 5, 6]
  int *d_values_out; // e.g., [-, -, -, -, -, -, -]

  num_items = 7;
  num_segments = 3;
  cudaMallocManaged(&d_offsets, (num_segments + 1) * sizeof(*d_offsets));
  cudaMallocManaged(&d_keys_in, num_items * sizeof(*d_keys_in));
  cudaMallocManaged(&d_keys_out, num_items * sizeof(*d_keys_out));
  cudaMallocManaged(&d_values_in, num_items * sizeof(*d_values_in));
  cudaMallocManaged(&d_values_out, num_items * sizeof(*d_values_out));

  d_offsets[0] = 0;
  d_offsets[1] = 3;
  d_offsets[2] = 3;
  d_offsets[3] = 7;

  d_keys_in[0] = 8;
  d_keys_in[1] = 6;
  d_keys_in[2] = 7;
  d_keys_in[3] = 5;
  d_keys_in[4] = 3;
  d_keys_in[5] = 0;
  d_keys_in[6] = 9;

  d_values_in[0] = 0;
  d_values_in[1] = 1;
  d_values_in[2] = 2;
  d_values_in[3] = 3;
  d_values_in[4] = 4;
  d_values_in[5] = 5;
  d_values_in[6] = 6;

  cub::DoubleBuffer<int> d_keys(d_keys_in, d_keys_out);
  cub::DoubleBuffer<int> d_values(d_values_in, d_values_out);

  cudaDeviceSynchronize();

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedSort::StableSortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items,
      num_segments, d_offsets, d_offsets + 1);

  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Run sorting operation
  cub::DeviceSegmentedSort::StableSortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items,
      num_segments, d_offsets, d_offsets + 1);

  // d_keys.Current()      <-- [8, 7, 6, 9, 5, 3, 0]
  // d_values.Current()    <-- [0, 2, 1, 6, 3, 4, 5]

  std::vector<int> d_keys_out_expected = {8, 7, 6, 9, 5, 3, 0};
  std::vector<int> d_values_out_expected = {0, 2, 1, 6, 3, 4, 5};

  cudaDeviceSynchronize();
  for (int i = 0; i < num_items; i++) {
    if (d_keys.Current()[i] != d_keys_out_expected[i] ||
        d_values.Current()[i] != d_values_out_expected[i]) {
      return false;
    }
  }

  return true;
}

int main() {
  if (!testStableSortPairs()) {
    std::cerr << "StableSortPairs test failed" << std::endl;
    return EXIT_FAILURE;
  }

  if (!testStableSortPairsDB()) {
    std::cerr << "StableSortPairs (DoubleBuffer) test failed" << std::endl;
    return EXIT_FAILURE;
  }

  if (!testStableSortPairsDescending()) {
    std::cerr << "StableSortPairsDescending test failed" << std::endl;
    return EXIT_FAILURE;
  }

  if (!testStableSortPairsDescendingDB()) {
    std::cerr << "StableSortPairsDescending (DoubleBuffer) test failed"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Tests passed" << std::endl;
  return EXIT_SUCCESS;
}
