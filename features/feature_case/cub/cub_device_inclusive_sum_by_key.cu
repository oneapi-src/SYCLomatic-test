#include <cub/cub.cuh>
#include <vector>
#include <stdio.h>

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
  std::vector<int> expected({8, 14, 7, 12, 15, 0, 9});
  // Determine temporary device storage requirements for inclusive prefix sum
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSumByKey(d_temp_storage, temp_storage_bytes,
                                     d_keys_in, d_values_in, d_values_out,
                                     num_items);
  // Allocate temporary storage for inclusive prefix sum
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run inclusive prefix sum
  cub::DeviceScan::InclusiveSumByKey(d_temp_storage, temp_storage_bytes,
                                     d_keys_in, d_values_in, d_values_out,
                                     num_items);
  std::vector<int> output(num_items, 0);
  cudaMemcpy(output.data(), d_values_out, sizeof(int) * num_items, cudaMemcpyDeviceToHost);
  cudaFree(d_keys_in);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_temp_storage);
  return std::equal(expected.begin(), expected.end(), output.begin());
}

int main() {
  if (!test()) {
    printf("cub::DeviceScan::InclusiveSumByKey test failed\n");
    return 1;
  }
  return 0;
}
