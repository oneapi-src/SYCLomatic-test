// ====------ thrust_is_sorted.cu---------- *- CUDA -* -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===--------------------------------------------------------------------===//

#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

void test_1() { // host iterator
  const int N = 6;
  int datas[N] = {1, 4, 2, 8, 5, 7};

  thrust::host_vector<int> v(datas, datas + N);

  bool result = thrust::is_sorted(thrust::host, v.begin(), v.end());
  if (result == true) {
    printf("test_1 run failed\n");
    exit(-1);
  }
  thrust::sort(v.begin(), v.end());
  result = thrust::is_sorted(thrust::host, v.begin(), v.end());
  if (result == false) {
    printf("test_1 run failed\n");
    exit(-1);
  }

  printf("test_1 run passed!\n");
}

void test_2() { // host iterator

  const int N = 6;
  int datas[N] = {1, 4, 2, 8, 5, 7};

  thrust::host_vector<int> v(datas, datas + N);

  bool result = thrust::is_sorted(v.begin(), v.end());
  if (result == true) {
    printf("test_2 run failed\n");
    exit(-1);
  }
  thrust::sort(v.begin(), v.end());
  result = thrust::is_sorted(v.begin(), v.end());
  if (result == false) {
    printf("test_2 run failed\n");
    exit(-1);
  }

  printf("test_2 run passed!\n");
}

void test_3() { // host iterator

  const int N = 6;
  int datas[N] = {1, 4, 2, 8, 5, 7};

  thrust::host_vector<int> v(datas, datas + N);
  thrust::greater<int> comp;
  bool result = thrust::is_sorted(thrust::host, v.begin(), v.end(), comp);
  if (result == true) {
    printf("test_3 run failed\n");
    exit(-1);
  }
  thrust::sort(v.begin(), v.end(), comp);
  result = thrust::is_sorted(thrust::host, v.begin(), v.end(), comp);
  if (result == false) {
    printf("test_3 run failed\n");
    exit(-1);
  }

  printf("test_3 run passed!\n");
}

void test_4() { // host iterator

  const int N = 6;
  int datas[N] = {1, 4, 2, 8, 5, 7};

  thrust::host_vector<int> v(datas, datas + N);
  thrust::greater<int> comp;
  bool result = thrust::is_sorted(v.begin(), v.end(), comp);
  if (result == true) {
    printf("test_4 run failed\n");
    exit(-1);
  }
  thrust::sort(v.begin(), v.end(), comp);
  result = thrust::is_sorted(v.begin(), v.end(), comp);
  if (result == false) {
    printf("test_4 run failed\n");
    exit(-1);
  }

  printf("test_4 run passed!\n");
}

void test_5() { // device iterator
  const int N = 6;
  int datas[N] = {1, 4, 2, 8, 5, 7};

  thrust::device_vector<int> v(datas, datas + N);

  bool result = thrust::is_sorted(thrust::device, v.begin(), v.end());
  if (result == true) {
    printf("test_5 run failed\n");
    exit(-1);
  }
  thrust::sort(v.begin(), v.end());
  result = thrust::is_sorted(thrust::device, v.begin(), v.end());
  if (result == false) {
    printf("test_5 run failed\n");
    exit(-1);
  }

  printf("test_5 run passed!\n");
}

void test_6() { // device iterator

  const int N = 6;
  int datas[N] = {1, 4, 2, 8, 5, 7};

  thrust::device_vector<int> v(datas, datas + N);

  bool result = thrust::is_sorted(v.begin(), v.end());
  if (result == true) {
    printf("test_6 run failed\n");
    exit(-1);
  }
  thrust::sort(v.begin(), v.end());
  result = thrust::is_sorted(v.begin(), v.end());
  if (result == false) {
    printf("test_6 run failed\n");
    exit(-1);
  }

  printf("test_6 run passed!\n");
}

void test_7() { // device iterator

  const int N = 6;
  int datas[N] = {1, 4, 2, 8, 5, 7};

  thrust::device_vector<int> v(datas, datas + N);
  thrust::greater<int> comp;
  bool result = thrust::is_sorted(thrust::device, v.begin(), v.end(), comp);
  if (result == true) {
    printf("test_7 run failed\n");
    exit(-1);
  }
  thrust::sort(v.begin(), v.end(), comp);
  result = thrust::is_sorted(thrust::device, v.begin(), v.end(), comp);
  if (result == false) {
    printf("test_7 run failed\n");
    exit(-1);
  }

  printf("test_7 run passed!\n");
}

void test_8() { // device iterator

  const int N = 6;
  int datas[N] = {1, 4, 2, 8, 5, 7};

  thrust::device_vector<int> v(datas, datas + N);
  thrust::greater<int> comp;
  bool result = thrust::is_sorted(v.begin(), v.end(), comp);
  if (result == true) {
    printf("test_8 run failed\n");
    exit(-1);
  }
  thrust::sort(v.begin(), v.end(), comp);
  result = thrust::is_sorted(v.begin(), v.end(), comp);
  if (result == false) {
    printf("test_8 run failed\n");
    exit(-1);
  }

  printf("test_8 run passed!\n");
}

int main() {
  test_1();
  test_2();
  test_3();
  test_4();
  test_5();
  test_6();
  test_7();
  test_8();

  return 0;
}