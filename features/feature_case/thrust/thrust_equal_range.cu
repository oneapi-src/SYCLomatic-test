// ====------ thrust_equal_range.cu------------- *- CUDA -*----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===-----------------------------------------------------------------------===//

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>

void test_1() {
  int data[] = {0, 2, 5, 7, 8};
  const int N = 5;

  thrust::device_vector<int> device_vec(data, data + N);

  typedef thrust::device_vector<int>::iterator Iterator;
  thrust::pair<Iterator, Iterator> result;
  result = thrust::equal_range(thrust::device, device_vec.begin(),
                               device_vec.end(), 0);
  if (*result.first != 0 || *result.second != 2) {
    std::cout << "test_1 run failed!\n";
    exit(-1);
  }

  printf("test_1 run passed!\n");
}

void test_2() {
  int data[] = {0, 2, 5, 7, 8};
  const int N = 5;

  thrust::device_vector<int> device_vec(data, data + N);

  typedef thrust::device_vector<int>::iterator Iterator;
  thrust::pair<Iterator, Iterator> result;
  result = thrust::equal_range(device_vec.begin(), device_vec.end(), 0);
  if (*result.first != 0 || *result.second != 2) {
    std::cout << "test_1 run failed!\n";
    exit(-1);
  }

  printf("test_2 run passed!\n");
}

void test_3() {
  int data[] = {0, 2, 5, 7, 8};
  const int N = 5;

  thrust::device_vector<int> device_vec(data, data + N);

  typedef thrust::device_vector<int>::iterator Iterator;
  thrust::pair<Iterator, Iterator> result;
  result = thrust::equal_range(thrust::device, device_vec.begin(),
                               device_vec.end(), 0, thrust::less<int>());
  if (*result.first != 0 || *result.second != 2) {
    std::cout << "test_1 run failed!\n";
    exit(-1);
  }

  printf("test_3 run passed!\n");
}

void test_4() {
  int data[] = {0, 2, 5, 7, 8};
  const int N = 5;

  thrust::device_vector<int> device_vec(data, data + N);

  typedef thrust::device_vector<int>::iterator Iterator;
  thrust::pair<Iterator, Iterator> result;
  result = thrust::equal_range(device_vec.begin(), device_vec.end(), 0,
                               thrust::less<int>());
  if (*result.first != 0 || *result.second != 2) {
    std::cout << "test_1 run failed!\n";
    exit(-1);
  }

  printf("test_4 run passed!\n");
}

void test_5() {
  int data[] = {0, 2, 5, 7, 8};
  const int N = 5;

  thrust::host_vector<int> host_vec(data, data + N);

  typedef thrust::host_vector<int>::iterator Iterator;
  thrust::pair<Iterator, Iterator> result;
  result =
      thrust::equal_range(thrust::host, host_vec.begin(), host_vec.end(), 0);
  if (*result.first != 0 || *result.second != 2) {
    std::cout << "test_1 run failed!\n";
    exit(-1);
  }

  printf("test_5 run passed!\n");
}

void test_6() {
  thrust::host_vector<int> input(5);
  int data[] = {0, 2, 5, 7, 8};
  const int N = 5;

  thrust::host_vector<int> host_vec(data, data + N);

  typedef thrust::host_vector<int>::iterator Iterator;
  thrust::pair<Iterator, Iterator> result;
  result = thrust::equal_range(host_vec.begin(), host_vec.end(), 0);
  if (*result.first != 0 || *result.second != 2) {
    std::cout << "test_1 run failed!\n";
    exit(-1);
  }

  printf("test_6 run passed!\n");
}

void test_7() {
  thrust::host_vector<int> input(5);
  int data[] = {0, 2, 5, 7, 8};
  const int N = 5;

  thrust::host_vector<int> host_vec(data, data + N);

  typedef thrust::host_vector<int>::iterator Iterator;
  thrust::pair<Iterator, Iterator> result;
  result = thrust::equal_range(thrust::host, host_vec.begin(), host_vec.end(),
                               0, thrust::less<int>());
  if (*result.first != 0 || *result.second != 2) {
    std::cout << "test_1 run failed!\n";
    exit(-1);
  }

  printf("test_7 run passed!\n");
}

void test_8() {
  thrust::host_vector<int> input(5);
  int data[] = {0, 2, 5, 7, 8};
  const int N = 5;

  thrust::host_vector<int> host_vec(data, data + N);

  typedef thrust::host_vector<int>::iterator Iterator;
  thrust::pair<Iterator, Iterator> result;
  result = thrust::equal_range(host_vec.begin(), host_vec.end(), 0,
                               thrust::less<int>());
  if (*result.first != 0 || *result.second != 2) {
    std::cout << "test_1 run failed!\n";
    exit(-1);
  }

  printf("test_8 run passed!\n");
}

void test_9() {
  int data[] = {0, 2, 5, 7, 8};
  const int N = 5;

  thrust::pair<int *, int *> result;
  result = thrust::equal_range(thrust::host, data, data + N, 0);
  if (*result.first != 0 || *result.second != 2) {
    std::cout << "test_1 run failed!\n";
    exit(-1);
  }

  printf("test_9 run passed!\n");
}

void test_10() {
  int data[] = {0, 2, 5, 7, 8};
  const int N = 5;

  thrust::pair<int *, int *> result;
  result = thrust::equal_range(data, data + N, 0);
  if (*result.first != 0 || *result.second != 2) {
    std::cout << "test_1 run failed!\n";
    exit(-1);
  }

  printf("test_10 run passed!\n");
}

void test_11() {
  int data[] = {0, 2, 5, 7, 8};
  const int N = 5;

  thrust::pair<int *, int *> result;
  result =
      thrust::equal_range(thrust::host, data, data + N, 0, thrust::less<int>());
  if (*result.first != 0 || *result.second != 2) {
    std::cout << "test_1 run failed!\n";
    exit(-1);
  }

  printf("test_11 run passed!\n");
}

void test_12() {
  int data[] = {0, 2, 5, 7, 8};
  const int N = 5;

  thrust::pair<int *, int *> result;
  result = thrust::equal_range(data, data + N, 0, thrust::less<int>());
  if (*result.first != 0 || *result.second != 2) {
    std::cout << "test_1 run failed!\n";
    exit(-1);
  }

  printf("test_12 run passed!\n");
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
  test_9();
  test_10();
  test_11();
  test_12();

  return 0;
}
