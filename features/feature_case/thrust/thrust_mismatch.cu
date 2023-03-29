// ====------ thrust_mismatch.cu----------------------------------------- *- CUDA -*
// -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/mismatch.h>

void test_1() {

  int data_1[] = {0, 5, 3, 7};
  int data_2[] = {0, 5, 9, 7};
  const int N = 4;

  thrust::device_vector<int> vec1(data_1, data_1 + N);
  thrust::device_vector<int> vec2(data_2, data_2 + N);

  typedef thrust::device_vector<int>::iterator Iterator;
  thrust::pair<Iterator, Iterator> result;
  result =
      thrust::mismatch(thrust::device, vec1.begin(), vec1.end(), vec2.begin());

  if (*result.first != 3 || *result.second != 9) {
    std::cout << "test_1 run failed!\n";
    exit(-1);
  }

  std::cout << "test_1 run passed!\n";
}

void test_2() {

  int data_1[] = {0, 5, 3, 7};
  int data_2[] = {0, 5, 9, 7};
  const int N = 4;

  thrust::host_vector<int> vec1(data_1, data_1 + N);
  thrust::host_vector<int> vec2(data_2, data_2 + N);

  typedef thrust::host_vector<int>::iterator Iterator;
  thrust::pair<Iterator, Iterator> result;
  result =
      thrust::mismatch(thrust::host, vec1.begin(), vec1.end(), vec2.begin());

  if (*result.first != 3 || *result.second != 9) {
    std::cout << "test_2 run failed!\n";
    exit(-1);
  }

  std::cout << "test_2 run passed!\n";
}

void test_3() {

  int data_1[] = {0, 5, 3, 7};
  int data_2[] = {0, 5, 9, 7};
  const int N = 4;

  typedef thrust::host_vector<int>::iterator Iterator;
  thrust::pair<int *, int *> result;
  result = thrust::mismatch(thrust::host, data_1, data_1 + N, data_2);

  if (*result.first != 3 || *result.second != 9) {
    std::cout << "test_3 run failed!\n";
    exit(-1);
  }

  std::cout << "test_3 run passed!\n";
}

void test_4() {

  int data_1[] = {0, 5, 3, 7};
  int data_2[] = {0, 5, 9, 7};
  const int N = 4;

  thrust::device_vector<int> vec1(data_1, data_1 + N);
  thrust::device_vector<int> vec2(data_2, data_2 + N);

  typedef thrust::device_vector<int>::iterator Iterator;
  thrust::pair<Iterator, Iterator> result;
  result = thrust::mismatch(vec1.begin(), vec1.end(), vec2.begin());

  if (*result.first != 3 || *result.second != 9) {
    std::cout << "test_4 run failed!\n";
    exit(-1);
  }

  std::cout << "test_4 run passed!\n";
}

void test_5() {

  int data_1[] = {0, 5, 3, 7};
  int data_2[] = {0, 5, 9, 7};
  const int N = 4;

  thrust::host_vector<int> vec1(data_1, data_1 + N);
  thrust::host_vector<int> vec2(data_2, data_2 + N);

  typedef thrust::host_vector<int>::iterator Iterator;
  thrust::pair<Iterator, Iterator> result;
  result = thrust::mismatch(vec1.begin(), vec1.end(), vec2.begin());

  if (*result.first != 3 || *result.second != 9) {
    std::cout << "test_5 run failed!\n";
    exit(-1);
  }

  std::cout << "test_5 run passed!\n";
}

void test_6() {

  int data_1[] = {0, 5, 3, 7};
  int data_2[] = {0, 5, 9, 7};
  const int N = 4;

  typedef thrust::host_vector<int>::iterator Iterator;
  thrust::pair<int *, int *> result;
  result = thrust::mismatch(data_1, data_1 + N, data_2);

  if (*result.first != 3 || *result.second != 9) {
    std::cout << "test_6 run failed!\n";
    exit(-1);
  }

  std::cout << "test_6 run passed!\n";
}

void test_7() {

  int data_1[] = {0, 5, 3, 7};
  int data_2[] = {0, 5, 9, 7};
  const int N = 4;

  thrust::device_vector<int> vec1(data_1, data_1 + N);
  thrust::device_vector<int> vec2(data_2, data_2 + N);

  typedef thrust::device_vector<int>::iterator Iterator;
  thrust::pair<Iterator, Iterator> result;
  result = thrust::mismatch(thrust::device, vec1.begin(), vec1.end(),
                            vec2.begin(), thrust::equal_to<int>());

  if (*result.first != 3 || *result.second != 9) {
    std::cout << "test_7 run failed!\n";
    exit(-1);
  }

  std::cout << "test_7 run passed!\n";
}

void test_8() {

  int data_1[] = {0, 5, 3, 7};
  int data_2[] = {0, 5, 9, 7};
  const int N = 4;

  thrust::host_vector<int> vec1(data_1, data_1 + N);
  thrust::host_vector<int> vec2(data_2, data_2 + N);

  typedef thrust::host_vector<int>::iterator Iterator;
  thrust::pair<Iterator, Iterator> result;
  result = thrust::mismatch(thrust::host, vec1.begin(), vec1.end(),
                            vec2.begin(), thrust::equal_to<int>());

  if (*result.first != 3 || *result.second != 9) {
    std::cout << "test_8 run failed!\n";
    exit(-1);
  }

  std::cout << "test_8 run passed!\n";
}

void test_9() {

  int data_1[] = {0, 5, 3, 7};
  int data_2[] = {0, 5, 9, 7};
  const int N = 4;

  typedef thrust::host_vector<int>::iterator Iterator;
  thrust::pair<int *, int *> result;
  result = thrust::mismatch(thrust::host, data_1, data_1 + N, data_2,
                            thrust::equal_to<int>());

  if (*result.first != 3 || *result.second != 9) {
    std::cout << "test_9 run failed!\n";
    exit(-1);
  }

  std::cout << "test_9 run passed!\n";
}

void test_10() {

  int data_1[] = {0, 5, 3, 7};
  int data_2[] = {0, 5, 9, 7};
  const int N = 4;

  thrust::device_vector<int> vec1(data_1, data_1 + N);
  thrust::device_vector<int> vec2(data_2, data_2 + N);

  typedef thrust::device_vector<int>::iterator Iterator;
  thrust::pair<Iterator, Iterator> result;
  result = thrust::mismatch(vec1.begin(), vec1.end(), vec2.begin(),
                            thrust::equal_to<int>());

  if (*result.first != 3 || *result.second != 9) {
    std::cout << "test_10 run failed!\n";
    exit(-1);
  }

  std::cout << "test_10 run passed!\n";
}

void test_11() {

  int data_1[] = {0, 5, 3, 7};
  int data_2[] = {0, 5, 9, 7};
  const int N = 4;

  thrust::host_vector<int> vec1(data_1, data_1 + N);
  thrust::host_vector<int> vec2(data_2, data_2 + N);

  typedef thrust::host_vector<int>::iterator Iterator;
  thrust::pair<Iterator, Iterator> result;
  result = thrust::mismatch(vec1.begin(), vec1.end(), vec2.begin(),
                            thrust::equal_to<int>());

  if (*result.first != 3 || *result.second != 9) {
    std::cout << "test_11 run failed!\n";
    exit(-1);
  }

  std::cout << "test_11 run passed!\n";
}

void test_12() {

  int data_1[] = {0, 5, 3, 7};
  int data_2[] = {0, 5, 9, 7};
  const int N = 4;

  typedef thrust::host_vector<int>::iterator Iterator;
  thrust::pair<int *, int *> result;
  result =
      thrust::mismatch(data_1, data_1 + N, data_2, thrust::equal_to<int>());

  if (*result.first != 3 || *result.second != 9) {
    std::cout << "test_12 run failed!\n";
    exit(-1);
  }

  std::cout << "test_12 run passed!\n";
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
