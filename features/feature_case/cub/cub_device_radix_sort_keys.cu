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

bool test(bool useDoubleBuffer=false) {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{0, 3, 5, 6, 7, 8, 9};
  cub::DoubleBuffer<int> buffers(d_keys_in, d_keys_out);

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  auto doSort = [&]() {
    if (useDoubleBuffer) {
      cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, buffers,
                                     num_items);
    } else {
      cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,
                                     d_keys_out, num_items);
    }
  };

  // Determine temporary device storage requirements
  doSort();
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  doSort();
  
  std::vector<int> keys_out(num_items);
  if (useDoubleBuffer) {
    cudaMemcpy(keys_out.data(), buffers.Current(), sizeof(int) * num_items,
               cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
               cudaMemcpyDeviceToHost);
  }
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin());
}

bool test1(bool useDoubleBuffer=false) {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{3, 0, 6, 7, 5, 8, 9};
  cub::DoubleBuffer<int> buffers(d_keys_in, d_keys_out);

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  auto doSort = [&]() {
    if (useDoubleBuffer) {
      cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, buffers,
                                     num_items, 2);
    } else {
      cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,
                                     d_keys_out, num_items, 2);
    }
  };

  // Determine temporary device storage requirements
  doSort();
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  doSort();
  
  std::vector<int> keys_out(num_items);
  if (useDoubleBuffer) {
    cudaMemcpy(keys_out.data(), buffers.Current(), sizeof(int) * num_items,
               cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
               cudaMemcpyDeviceToHost);
  }
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin());
}

bool test2(bool useDoubleBuffer=false) {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{3, 0, 6, 7, 5, 8, 9};
  cub::DoubleBuffer<int> buffers(d_keys_in, d_keys_out);

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  auto doSort = [&]() {
    if (useDoubleBuffer) {
      cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, buffers,
                                     num_items, 2, 4);
    } else {
      cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,
                                     d_keys_out, num_items, 2, 4);
    }
  };
  
  // Determine temporary device storage requirements
  doSort();
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  doSort();
  
  std::vector<int> keys_out(num_items);
  cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
             cudaMemcpyDeviceToHost);
  if (useDoubleBuffer) {
    cudaMemcpy(keys_out.data(), buffers.Current(), sizeof(int) * num_items,
               cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
               cudaMemcpyDeviceToHost);
  }
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin());
}

bool test3(bool useDoubleBuffer=false) {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{3, 0, 6, 7, 5, 8, 9};
  cub::DoubleBuffer<int> buffers(d_keys_in, d_keys_out);
  cudaStream_t s = nullptr;
  cudaStreamCreate(&s);

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  auto doSort = [&]() {
    if (useDoubleBuffer) {
      cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, buffers,
                                     num_items, 2, 4, s);
    } else {
      cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in,
                                     d_keys_out, num_items, 2, 4, s);
    }
  };
  // Determine temporary device storage requirements
  doSort();
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  doSort();

  std::vector<int> keys_out(num_items);
  if (useDoubleBuffer) {
    cudaMemcpy(keys_out.data(), buffers.Current(), sizeof(int) * num_items,
               cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
               cudaMemcpyDeviceToHost);
  }
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  cudaStreamDestroy(s);
  
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin());
}

bool test_descending(bool useDoubleBuffer=false) {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{9, 8, 7, 6, 5, 3, 0};
  cub::DoubleBuffer<int> buffers(d_keys_in, d_keys_out);

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  auto doSort = [&]() {
    if (useDoubleBuffer) {
      cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes,
                                               buffers, num_items);
    } else {
      cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes,
                                               d_keys_in, d_keys_out, num_items);
    }
  };

  // Determine temporary device storage requirements
  doSort();
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  doSort();

  std::vector<int> keys_out(num_items);
  if (useDoubleBuffer) {
    cudaMemcpy(keys_out.data(), buffers.Current(), sizeof(int) * num_items,
               cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
               cudaMemcpyDeviceToHost);
  }
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin());
}

bool test_descending1(bool useDoubleBuffer=false) {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{8, 9, 6, 7, 5, 3, 0};
  cub::DoubleBuffer<int> buffers(d_keys_in, d_keys_out);

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  auto doSort = [&]() {
    if (useDoubleBuffer) {
      cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes,
                                               buffers, num_items, 2);
    } else {
      cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes,
                                               d_keys_in, d_keys_out, num_items, 2);
    }
  };
  // Determine temporary device storage requirements
  doSort();
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  doSort();

  std::vector<int> keys_out(num_items);
  if (useDoubleBuffer) {
    cudaMemcpy(keys_out.data(), buffers.Current(), sizeof(int) * num_items,
               cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
               cudaMemcpyDeviceToHost);
  }
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin());
}

bool test_descending2(bool useDoubleBuffer=false) {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{8, 9, 6, 7, 5, 3, 0};
  cub::DoubleBuffer<int> buffers(d_keys_in, d_keys_out);

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  auto doSort = [&]() {
    if (useDoubleBuffer) {
      cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes,
                                               buffers, num_items, 2, 4);
    } else {
      cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes,
                                               d_keys_in, d_keys_out, num_items, 2, 4);
    }
  };

  // Determine temporary device storage requirements
  doSort();
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  doSort();

  std::vector<int> keys_out(num_items);
  if (useDoubleBuffer) {
    cudaMemcpy(keys_out.data(), buffers.Current(), sizeof(int) * num_items,
               cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
               cudaMemcpyDeviceToHost);
  }
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin());
}

bool test_descending3(bool useDoubleBuffer=false) {
  int num_items = 7;
  int *d_keys_in = init({8, 6, 7, 5, 3, 0, 9});
  int *d_keys_out = init({0, 0, 0, 0, 0, 0, 0});
  std::vector<int> expected_keys_out{8, 9, 6, 7, 5, 3, 0};
  cudaStream_t s = nullptr;
  cudaStreamCreate(&s);
  cub::DoubleBuffer<int> buffers(d_keys_in, d_keys_out);

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  auto doSort = [&]() {
    if (useDoubleBuffer) {
      cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes,
                                               buffers, num_items, 2, 4, s);
    } else {
      cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes,
                                               d_keys_in, d_keys_out, num_items, 2, 4, s);
    }
  };
  // Determine temporary device storage requirements
  doSort();
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  doSort();

  std::vector<int> keys_out(num_items);
  if (useDoubleBuffer) {
    cudaMemcpy(keys_out.data(), buffers.Current(), sizeof(int) * num_items,
               cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(keys_out.data(), d_keys_out, sizeof(int) * num_items,
               cudaMemcpyDeviceToHost);
  }
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(d_temp_storage);
  cudaStreamDestroy(s);
  
  return std::equal(expected_keys_out.begin(), expected_keys_out.end(),
                    keys_out.begin());
}

int main() {
  int res = 0;
  for (auto b : {false, true}) {
    auto s = b ? " with double buffer" : "";
    if (!test(b)) {
      printf("cub::DeviceRadixSort::SortKeys%s failed\n", s);
      res = 1;
    }
    if (!test1(b)) {
      printf("cub::DeviceRadixSort::SortKeys%s failed\n", s);
      res = 1;
    }
    if (!test2(b)) {
      printf("cub::DeviceRadixSort::SortKeys%s failed\n", s);
      res = 1;
    }
    if (!test3(b)) {
      printf("cub::DeviceRadixSort::SortKeys%s failed\n", s);
      res = 1;
    }

    if (!test_descending(b)) {
      printf("cub::DeviceRadixSort::SortKeysDescending%s failed\n", s);
      res = 1;
    }

    if (!test_descending1(b)) {
      printf("cub::DeviceRadixSort::SortKeysDescending%s failed\n", s);
      res = 1;
    }

    if (!test_descending2(b)) {
      printf("cub::DeviceRadixSort::SortKeysDescending%s failed\n", s);
      res = 1;
    }

    if (!test_descending3(b)) {
      printf("cub::DeviceRadixSort::SortKeysDescending%s failed\n", s);
      res = 1;
    }
  }

  return res;
}
