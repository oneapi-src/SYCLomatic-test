// ====------ thrust_is_partitioned.cu--------------- *- CUDA -*-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//


#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>

struct is_even {
  __host__ __device__ bool operator()(const int &x) const { return (x % 2) == 0; }
};

void test_1() {

  int A[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  int B[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  bool result = thrust::is_partitioned(thrust::host, A, A + 10,
                                       is_even()); 
  if (!result) {
    printf("test_1 1 run failed\n");
    exit(-1);
  }

  result = thrust::is_partitioned(thrust::host, B, B + 10,
                                  is_even()); 
  if (result) {
    printf("test_1 2 run failed\n");
    exit(-1);
  }

  printf("test_1 run passed!\n");
}

void test_2() {

  int A[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  int B[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  bool result = thrust::is_partitioned(A, A + 10, is_even()); 
  if (!result) {
    printf("test_2 1 run failed\n");
    exit(-1);
  }

  result = thrust::is_partitioned(B, B + 10, is_even()); 
  if (result) {
    printf("test_2 2 run failed\n");
    exit(-1);
  }

  printf("test_2 run passed!\n");
}

void test_3() {

  int A[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  int B[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  thrust::host_vector<int> h_A(A, A + 10);
  thrust::host_vector<int> h_B(B, B + 10);

  bool result = thrust::is_partitioned(thrust::host, h_A.begin(), h_A.end(),
                                       is_even()); 
  if (!result) {
    printf("test_3 1 run failed\n");
    exit(-1);
  }

  result = thrust::is_partitioned(thrust::host, h_B.begin(), h_B.end(),
                                  is_even()); 
  if (result) {
    printf("test_3 2 run failed\n");
    exit(-1);
  }

  printf("test_3 run passed!\n");
}

void test_4() {

  int A[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  int B[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  thrust::host_vector<int> h_A(A, A + 10);
  thrust::host_vector<int> h_B(B, B + 10);

  bool result =
      thrust::is_partitioned(h_A.begin(), h_A.end(), is_even()); 
  if (!result) {
    printf("test_4 1 run failed\n");
    exit(-1);
  }

  result = thrust::is_partitioned(h_B.begin(), h_B.end(),
                                  is_even()); 
  if (result) {
    printf("test_4 2 run failed\n");
    exit(-1);
  }

  printf("test_4 run passed!\n");
}

void test_5() {

  int A[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  int B[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  thrust::device_vector<int> d_A(A, A + 10);
  thrust::device_vector<int> d_B(B, B + 10);

  bool result = thrust::is_partitioned(thrust::device, d_A.begin(), d_A.end(),
                                       is_even()); 
  if (!result) {
    printf("test_5 1 run failed\n");
    exit(-1);
  }

  result = thrust::is_partitioned(thrust::device, d_B.begin(), d_B.end(),
                                  is_even()); 
  if (result) {
    printf("test_5 2 run failed\n");
    exit(-1);
  }

  printf("test_5 run passed!\n");
}

void test_6() {

  int A[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  int B[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  thrust::device_vector<int> d_A(A, A + 10);
  thrust::device_vector<int> d_B(B, B + 10);

  bool result =
      thrust::is_partitioned(d_A.begin(), d_A.end(), is_even()); 
  if (!result) {
    printf("test_6 1 run failed\n");
    exit(-1);
  }

  result = thrust::is_partitioned(d_B.begin(), d_B.end(),
                                  is_even()); 
  if (result) {
    printf("test_6 2 run failed\n");
    exit(-1);
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