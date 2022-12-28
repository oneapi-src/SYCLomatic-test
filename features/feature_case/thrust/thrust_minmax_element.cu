// ====------ thrust_minmax_element.cu---------- *- CUDA -* -------------------===//
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
#include <thrust/unique.h>

struct key_value {
  int key;
  int value;
  __host__ __device__ bool operator!=(struct key_value &tmp) const {
    if (this->key != tmp.key || this->value != tmp.value) {
      return true;
    } else {
      return false;
    }
  }
};

struct compare_key_value {
  __host__ __device__ bool operator()(key_value lhs, key_value rhs) const {
    return lhs.key < rhs.key;
  }
};

void test_1() { // host iterator

  const int N = 6;
  int data[N] = {1, 0, 2, 2, 1, 3};

  thrust::host_vector<int> h_values(data, data + N);
  typedef thrust::pair<thrust::host_vector<int>::iterator,
                       thrust::host_vector<int>::iterator>
      iter_pair;
  iter_pair result =
      thrust::minmax_element(thrust::host, h_values.begin(), h_values.end());

  if (result.first != h_values.begin() + 1) {
    printf("test_1 min run failed\n");
    exit(-1);
  }
  if (result.second != h_values.begin() + 5) {
    printf("test_1 max run failed\n");
    exit(-1);
  }

  printf("test_1 run passed!\n");
}

void test_2() { // host iterator
  const int N = 6;
  int data[N] = {1, 0, 2, 2, 1, 3};

  thrust::host_vector<int> h_values(data, data + N);
  typedef thrust::pair<thrust::host_vector<int>::iterator,
                       thrust::host_vector<int>::iterator>
      iter_pair;
  iter_pair result = thrust::minmax_element(h_values.begin(), h_values.end());

  if (result.first != h_values.begin() + 1) {
    printf("test_2 min run failed\n");
    exit(-1);
  }
  if (result.second != h_values.begin() + 5) {
    printf("test_2 max run failed\n");
    exit(-1);
  }

  printf("test_2 run passed!\n");
}

void test_3() { // host iterator

  const int N = 4;
  key_value data[N] = {{4, 5}, {0, 7}, {2, 3}, {6, 1}};

  thrust::host_vector<key_value> h_values(data, data + N);
  typedef thrust::pair<thrust::host_vector<key_value>::iterator,
                       thrust::host_vector<key_value>::iterator>
      iter_pair;
  iter_pair result =
      thrust::minmax_element(thrust::host, h_values.begin(),
                             h_values.begin() + 4, compare_key_value());

  if (*result.first != *(h_values.begin() + 1)) {
    printf("test_3 min run failed\n");
    exit(-1);
  }
  if (*result.second != *(h_values.begin() + 3)) {
    printf("test_3 max run failed\n");
    exit(-1);
  }

  printf("test_3 run passed!\n");
}

void test_4() { // host iterator

  const int N = 4;
  key_value data[N] = {{4, 5}, {0, 7}, {2, 3}, {6, 1}};

  thrust::host_vector<key_value> h_values(data, data + N);
  typedef thrust::pair<thrust::host_vector<key_value>::iterator,
                       thrust::host_vector<key_value>::iterator>
      iter_pair;
  iter_pair result = thrust::minmax_element(
      h_values.begin(), h_values.begin() + 4, compare_key_value());

  if (*result.first != *(h_values.begin() + 1)) {
    printf("test_4 min run failed\n");
    exit(-1);
  }
  if (*result.second != *(h_values.begin() + 3)) {
    printf("test_4 max run failed\n");
    exit(-1);
  }

  printf("test_4 run passed!\n");
}

void test_5() { // device iterator

  const int N = 6;
  int data[N] = {1, 0, 2, 2, 1, 3};

  thrust::device_vector<int> d_values(data, data + N);
  typedef thrust::pair<thrust::device_vector<int>::iterator,
                       thrust::device_vector<int>::iterator>
      iter_pair;
  iter_pair result =
      thrust::minmax_element(thrust::device, d_values.begin(), d_values.end());

  if (result.first != d_values.begin() + 1) {
    printf("test_5 min run failed\n");
    exit(-1);
  }
  if (result.second != d_values.begin() + 5) {
    printf("test_5 max run failed\n");
    exit(-1);
  }

  printf("test_5 run passed!\n");
}

void test_6() { // device iterator
  const int N = 6;
  int data[N] = {1, 0, 2, 2, 1, 3};

  thrust::device_vector<int> d_values(data, data + N);
  typedef thrust::pair<thrust::device_vector<int>::iterator,
                       thrust::device_vector<int>::iterator>
      iter_pair;
  iter_pair result = thrust::minmax_element(d_values.begin(), d_values.end());

  if (result.first != d_values.begin() + 1) {
    printf("test_6 run failed!\n");

    exit(-1);
  }
  if (result.second != d_values.begin() + 5) {
    printf("test_6 run failed!\n");
    exit(-1);
  }
  printf("test_6 run passed!\n");
}

void test_7() { // device iterator

  const int N = 4;
  key_value data[N] = {{4, 5}, {0, 7}, {2, 3}, {6, 1}};

  thrust::device_vector<key_value> d_values(data, data + N);
  typedef thrust::pair<thrust::device_vector<key_value>::iterator,
                       thrust::device_vector<key_value>::iterator>
      iter_pair;
  iter_pair result = thrust::minmax_element(
      thrust::device, d_values.begin(), d_values.end(), compare_key_value());

  if (result.first != (d_values.begin() + 1)) {
    printf("test_7 min run failed\n");
    exit(-1);
  }
  if (result.second != (d_values.begin() + 3)) {
    printf("test_7 max run failed\n");
    exit(-1);
  }

  printf("test_7 run passed!\n");
}

void test_8() { // device iterator

  const int N = 4;
  key_value data[N] = {{4, 5}, {0, 7}, {2, 3}, {6, 1}};

  thrust::device_vector<key_value> d_values(data, data + N);
  typedef thrust::pair<thrust::device_vector<key_value>::iterator,
                       thrust::device_vector<key_value>::iterator>
      iter_pair;
  iter_pair result = thrust::minmax_element(d_values.begin(), d_values.end(),
                                            compare_key_value());

  if (result.first != (d_values.begin() + 1)) {
    printf("test_8 min run failed\n");
    exit(-1);
  }
  if (result.second != (d_values.begin() + 3)) {
    printf("test_8 max run failed\n");
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
