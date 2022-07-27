// ====------ thrust-find.cu ----------------------------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>

#include <iterator>

void test_1() {
  int values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  thrust::device_vector<int> d_values(values, values + 10);
  auto ret_iter = thrust::find(d_values.begin(), d_values.end(), 5);
  int d = std::distance(d_values.begin(), ret_iter);
  int d_ref = 4;
  if (d != d_ref) {
    printf("test_1 run failed\n");
    exit(-1);
  }
  printf("test_1 run passed!\n");
}

void test_2() {
  int values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  thrust::device_vector<int> d_values(values, values + 10);
  auto ret_iter = thrust::find(thrust::device, d_values.begin(), d_values.end(), 5);
  int d = std::distance(d_values.begin(), ret_iter);
  int d_ref = 4;
  if (d != d_ref) {
    printf("test_2 run failed\n");
    exit(-1);
  }
  printf("test_2 run passed!\n");
}

int main() {
  test_1();
  test_2();
  return 0;
}
