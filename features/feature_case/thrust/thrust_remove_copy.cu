// ====------ remove_copy.cu------------- *- CUDA -* -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===-------------------------------------------------------------------===//

#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

void test_1() { // host iterator
  const int N = 6;
  int A[N] = {-2, 0, -1, 0, 1, 2};
  int B[N - 2];
  int ans[N - 2] = {-2, -1, 1, 2};
  thrust::host_vector<int> V(A, A + N);
  thrust::host_vector<int> result(B, B + N - 2);

  thrust::remove_copy(thrust::host, V.begin(), V.end(), result.begin(), 0);
  for (int i = 0; i < N - 2; i++) {
    if (result[i] != ans[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }

  printf("test_1 run passed!\n");
}

void test_2() { // host iterator

  const int N = 6;
  int A[N] = {-2, 0, -1, 0, 1, 2};
  int B[N - 2];
  int ans[N - 2] = {-2, -1, 1, 2};
  thrust::host_vector<int> V(A, A + N);
  thrust::host_vector<int> result(B, B + N - 2);

  thrust::remove_copy(V.begin(), V.end(), result.begin(), 0);
  for (int i = 0; i < N - 2; i++) {
    if (result[i] != ans[i]) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }

  printf("test_2 run passed!\n");
}

void test_3() { // device iterator
  const int N = 6;
  int A[N] = {-2, 0, -1, 0, 1, 2};
  int B[N - 2];
  int ans[N - 2] = {-2, -1, 1, 2};
  thrust::device_vector<int> V(A, A + N);
  thrust::device_vector<int> result(B, B + N - 2);

  thrust::remove_copy(thrust::device, V.begin(), V.end(), result.begin(), 0);
  for (int i = 0; i < N - 2; i++) {
    if (result[i] != ans[i]) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }

  printf("test_3 run passed!\n");
}

void test_4() { // host iterator

  const int N = 6;
  int A[N] = {-2, 0, -1, 0, 1, 2};
  int B[N - 2];
  int ans[N - 2] = {-2, -1, 1, 2};
  thrust::device_vector<int> V(A, A + N);
  thrust::device_vector<int> result(B, B + N - 2);

  thrust::remove_copy(V.begin(), V.end(), result.begin(), 0);
  for (int i = 0; i < N - 2; i++) {
    if (result[i] != ans[i]) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }

  printf("test_4 run passed!\n");
}

void test_5() { // host iterator
  const int N = 6;
  int V[N] = {-2, 0, -1, 0, 1, 2};
  int result[N - 2];
  int ans[N - 2] = {-2, -1, 1, 2};

  thrust::remove_copy(thrust::host, V, V + N, result, 0);
  for (int i = 0; i < N - 2; i++) {
    if (result[i] != ans[i]) {
      printf("test_5 run failed\n");
      exit(-1);
    }
  }

  printf("test_5 run passed!\n");
}

void test_6() {

  const int N = 6;
  int V[N] = {-2, 0, -1, 0, 1, 2};
  int result[N - 2];
  int ans[N - 2] = {-2, -1, 1, 2};

  thrust::remove_copy(V, V + N, result, 0);
  for (int i = 0; i < N - 2; i++) {
    if (result[i] != ans[i]) {
      printf("test_6 run failed\n");
      exit(-1);
    }
  }

  printf("test_6 run passed!\n");
}

int main() {
  test_1();
  test_2();
  test_3(); // test_3 run failed when migrated with none-USM mode
  test_4();
  test_5();
  test_6();

  return 0;
}