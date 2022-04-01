// ====------ thrust-gather.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>

void test_1() {
  int values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  thrust::device_vector<int> d_values(values, values + 10);
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::device_vector<int> d_map(map, map + 10);
  thrust::device_vector<int> d_output(10);

  thrust::gather(d_map.begin(), d_map.end(), d_values.begin(),
                 d_output.begin());

  int values_ref[10] = {0, 9, 8, 7, 6, 1, 2, 3, 4, 5};
  for (int i = 0; i < d_output.size(); i++) {
    if (d_output[i] != values_ref[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }

  printf("test_1 run passed!\n");
}

void test_2() {
  int values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  thrust::device_vector<int> d_values(values, values + 10);
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::device_vector<int> d_map(map, map + 10);
  thrust::device_vector<int> d_output(10);
  thrust::gather(thrust::device, d_map.begin(), d_map.end(), d_values.begin(),
                 d_output.begin());

  int values_ref[10] = {0, 9, 8, 7, 6, 1, 2, 3, 4, 5};

  for (int i = 0; i < d_output.size(); i++) {
    if (d_output[i] != values_ref[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }
  printf("test_2 run passed!\n");
}

void test_3() {
  int values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  thrust::host_vector<int> h_values(values, values + 10);
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::host_vector<int> h_map(map, map + 10);
  thrust::host_vector<int> h_output(10);
  thrust::gather(thrust::seq, h_map.begin(), h_map.end(), h_values.begin(),
                 h_output.begin());

  int values_ref[10] = {0, 9, 8, 7, 6, 1, 2, 3, 4, 5};

  for (int i = 0; i < h_output.size(); i++) {
    if (h_output[i] != values_ref[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }
  printf("test_3 run passed!\n");
}

void test_4() {
  int values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  thrust::host_vector<int> h_values(values, values + 10);
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::host_vector<int> h_map(map, map + 10);
  thrust::host_vector<int> h_output(10);
  thrust::gather(h_map.begin(), h_map.end(), h_values.begin(),
                 h_output.begin());

  int values_ref[10] = {0, 9, 8, 7, 6, 1, 2, 3, 4, 5};

  for (int i = 0; i < h_output.size(); i++) {
    if (h_output[i] != values_ref[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }
  printf("test_4 run passed!\n");
}

int main() {

  test_1();
  test_2();
  test_3();
  test_4();

  return 0;
}
