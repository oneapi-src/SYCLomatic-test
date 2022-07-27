// ====------ thrust-inner_product.cu -------------------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

void test_1() {
  int vec_1[2] = {1, 2};
  int vec_2[2] = {3, 4};
  thrust::device_vector<int> vec_1_d(vec_1, vec_1 + 2);
  thrust::device_vector<int> vec_2_d(vec_2, vec_2 + 2);
  int ret = thrust::inner_product(thrust::device, vec_1_d.begin(), vec_1_d.end(), vec_2_d.begin(), 123);
  int ref = 134;
  if (ret != ref) {
    printf("test_1 run failed\n");
    exit(-1);
  }
  printf("test_1 run passed!\n");
}

void test_2() {
  int vec_1[2] = {1, 2};
  int vec_2[2] = {3, 4};
  thrust::device_vector<int> vec_1_d(vec_1, vec_1 + 2);
  thrust::device_vector<int> vec_2_d(vec_2, vec_2 + 2);
  int ret = thrust::inner_product(vec_1_d.begin(), vec_1_d.end(), vec_2_d.begin(), 123);
  int ref = 134;
  if (ret != ref) {
    printf("test_2 run failed\n");
    exit(-1);
  }
  printf("test_2 run passed!\n");
}

void test_3() {
  int vec_1[2] = {3, 2};
  int vec_2[2] = {3, 3};
  thrust::device_vector<int> vec_1_d(vec_1, vec_1 + 2);
  thrust::device_vector<int> vec_2_d(vec_2, vec_2 + 2);
  int ret = thrust::inner_product(thrust::device, vec_1_d.begin(), vec_1_d.end(), vec_2_d.begin(), 123, thrust::plus<int>(), thrust::not_equal_to<int>());
  int ref = 124;
  if (ret != ref) {
    printf("test_3 run failed\n");
    exit(-1);
  }
  printf("test_3 run passed!\n");
}

void test_4() {
  int vec_1[2] = {3, 2};
  int vec_2[2] = {3, 3};
  thrust::device_vector<int> vec_1_d(vec_1, vec_1 + 2);
  thrust::device_vector<int> vec_2_d(vec_2, vec_2 + 2);
  int ret = thrust::inner_product(vec_1_d.begin(), vec_1_d.end(), vec_2_d.begin(), 123, thrust::plus<int>(), thrust::not_equal_to<int>());
  int ref = 124;
  if (ret != ref) {
    printf("test_4 run failed\n");
    exit(-1);
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
