// ====------ thrust_uninitialized_fill_n.cu------------- *- CUDA -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/uninitialized_fill.h>

struct Int {
  __host__ __device__ Int(int x) : val(x) {}
  int val;
};

void test_1() {
  const int N = 137;
  Int val(46);
  thrust::device_ptr<Int> d_array = thrust::device_malloc<Int>(N);
  thrust::uninitialized_fill_n(d_array, N, val);

  thrust::host_vector<Int> hostVec(d_array, d_array + N);

  for (int i = 0; i < N; i++) {
    if (hostVec[i].val != 46) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }

  printf("test_1 run passed!\n");
}

void test_2() {

  const int N = 137;
  Int val(46);
  thrust::device_ptr<Int> d_array = thrust::device_malloc<Int>(N);
  thrust::uninitialized_fill_n(thrust::device, d_array, N, val);

  thrust::host_vector<Int> hostVec(d_array, d_array + N);

  for (int i = 0; i < N; i++) {
    if (hostVec[i].val != 46) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }

  printf("test_2 run passed!\n");
}

void test_3() {

  const int N = 137;
  int data[N];
  int val = 46;

  for (int i = 0; i < N; i++)
    data[i] = -1;

  thrust::uninitialized_fill_n(data, N, val);

  for (int i = 0; i < N; i++) {
    if (data[i] != 46) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }

  printf("test_3 run passed!\n");
}

void test_4() {

  const int N = 137;
  int data[N];
  int val = 46;

  for (int i = 0; i < N; i++)
    data[i] = -1;

  thrust::uninitialized_fill_n(thrust::host, data, N, val);

  for (int i = 0; i < N; i++) {
    if (data[i] != 46) {
      printf("test_4 run failed\n");
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