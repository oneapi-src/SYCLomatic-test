// ====------ thrust_transform_inclusive_scan.cu-- *- CUDA -*--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/transform_scan.h>

void test_1() {

  const int N = 6;
  int data[6] = {1, 0, 2, 2, 1, 3};
  int ans[N] = {-1, -1, -3, -5, -6, -9};

  thrust::negate<int> unary_op;
  thrust::plus<int> binary_op;
  thrust::transform_inclusive_scan(data, data + N, data, unary_op, binary_op);

  for (int i = 0; i < N; i++) {
    if (data[i] != ans[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }

  printf("test_1 run passed!\n");
}

void test_2() {

  const int N = 6;
  int data[6] = {1, 0, 2, 2, 1, 3};
  int ans[N] = {-1, -1, -3, -5, -6, -9};

  thrust::host_vector<int> h_vec_data(data, data + N);

  thrust::negate<int> unary_op;
  thrust::plus<int> binary_op;
  thrust::transform_inclusive_scan(h_vec_data.begin(), h_vec_data.end(),
                                   h_vec_data.begin(), unary_op, binary_op);

  for (int i = 0; i < N; i++) {
    if (h_vec_data[i] != ans[i]) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }

  printf("test_2 run passed!\n");
}

void test_3() {

  const int N = 6;
  int data[6] = {1, 0, 2, 2, 1, 3};
  int ans[N] = {-1, -1, -3, -5, -6, -9};

  thrust::device_vector<int> d_vec_data(data, data + N);

  thrust::negate<int> unary_op;
  thrust::plus<int> binary_op;
  thrust::transform_inclusive_scan(d_vec_data.begin(), d_vec_data.end(),
                                   d_vec_data.begin(), unary_op, binary_op);

  for (int i = 0; i < N; i++) {
    if (d_vec_data[i] != ans[i]) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }

  printf("test_3 run passed!\n");
}

void test_4() {

  const int N = 6;
  int data[6] = {1, 0, 2, 2, 1, 3};
  int ans[N] = {-1, -1, -3, -5, -6, -9};

  thrust::negate<int> unary_op;
  thrust::plus<int> binary_op;
  thrust::transform_inclusive_scan(thrust::host, data, data + N, data, unary_op,
                                   binary_op);

  for (int i = 0; i < N; i++) {
    if (data[i] != ans[i]) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }

  printf("test_4 run passed!\n");
}

void test_5() {

  const int N = 6;
  int data[6] = {1, 0, 2, 2, 1, 3};
  int ans[N] = {-1, -1, -3, -5, -6, -9};

  thrust::host_vector<int> h_vec_data(data, data + N);

  thrust::negate<int> unary_op;
  thrust::plus<int> binary_op;
  thrust::transform_inclusive_scan(thrust::host, h_vec_data.begin(),
                                   h_vec_data.end(), h_vec_data.begin(),
                                   unary_op, binary_op);

  for (int i = 0; i < N; i++) {
    if (h_vec_data[i] != ans[i]) {
      printf("test_5 run failed\n");
      exit(-1);
    }
  }

  printf("test_5 run passed!\n");
}

void test_6() {

  const int N = 6;
  int data[6] = {1, 0, 2, 2, 1, 3};
  int ans[N] = {-1, -1, -3, -5, -6, -9};

  thrust::device_vector<int> d_vec_data(data, data + N);

  thrust::negate<int> unary_op;
  thrust::plus<int> binary_op;
  thrust::transform_inclusive_scan(thrust::device, d_vec_data.begin(),
                                   d_vec_data.end(), d_vec_data.begin(),
                                   unary_op, binary_op);

  for (int i = 0; i < N; i++) {
    if (d_vec_data[i] != ans[i]) {
      printf("test_6 run failed\n");
      exit(-1);
    }
  }

  printf("test_6 run passed!\n");
}

int main() {

  test_1();
  test_2();
  test_3();
  test_4();
  test_5();
  test_6();

  return 0;
}
