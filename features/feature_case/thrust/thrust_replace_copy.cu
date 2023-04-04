// ====------ thrust_replace_copy.cu----------------------- *- CUDA -*
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/replace.h>

void test_1() {
  const int N = 4;
  int data[] = {1, 2, 3, 1};

  thrust::device_vector<int> d_data(data, data + N);
  thrust::device_vector<int> d_result(4);
  thrust::replace_copy(thrust::device, d_data.begin(), d_data.end(),
                       d_result.begin(), 1, 99);

  int ref[] = {99, 2, 3, 99};

  for (int i = 0; i < N; i++) {
    if (d_result[i] != ref[i]) {
      std::cout << "test_1 run failed!\n";
      exit(-1);
    }
  }

  std::cout << "test_1 run passed!\n";
}

void test_2() {
  const int N = 4;
  int data[] = {1, 2, 3, 1};

  thrust::host_vector<int> h_data(data, data + N);
  thrust::host_vector<int> h_result(4);
  thrust::replace_copy(thrust::host, h_data.begin(), h_data.end(),
                       h_result.begin(), 1, 99);

  int ref[] = {99, 2, 3, 99};

  for (int i = 0; i < N; i++) {
    if (h_result[i] != ref[i]) {
      std::cout << "test_2 run failed!\n";
      exit(-1);
    }
  }

  std::cout << "test_2 run passed!\n";
}

void test_3() {
  const int N = 4;
  int data[] = {1, 2, 3, 1};
  int result[N];

  thrust::replace_copy(thrust::host, data, data + N, result, 1, 99);

  int ref[] = {99, 2, 3, 99};

  for (int i = 0; i < N; i++) {
    if (result[i] != ref[i]) {
      std::cout << "test_3 run failed!\n";
      exit(-1);
    }
  }

  std::cout << "test_3 run passed!\n";
}

void test_4() {
  const int N = 4;
  int data[] = {1, 2, 3, 1};

  thrust::device_vector<int> d_data(data, data + N);
  thrust::device_vector<int> d_result(4);
  thrust::replace_copy(d_data.begin(), d_data.end(), d_result.begin(), 1, 99);

  int ref[] = {99, 2, 3, 99};

  for (int i = 0; i < N; i++) {
    if (d_result[i] != ref[i]) {
      std::cout << "test_4 run failed!\n";
      exit(-1);
    }
  }

  std::cout << "test_4 run passed!\n";
}

void test_5() {
  const int N = 4;
  int data[] = {1, 2, 3, 1};

  thrust::host_vector<int> h_data(data, data + N);
  thrust::host_vector<int> h_result(4);
  thrust::replace_copy(h_data.begin(), h_data.end(), h_result.begin(), 1, 99);

  int ref[] = {99, 2, 3, 99};

  for (int i = 0; i < N; i++) {
    if (h_result[i] != ref[i]) {
      std::cout << "test_5 run failed!\n";
      exit(-1);
    }
  }

  std::cout << "test_5 run passed!\n";
}

void test_6() {
  const int N = 4;
  int data[] = {1, 2, 3, 1};
  int result[N];

  thrust::replace_copy(data, data + N, result, 1, 99);

  int ref[] = {99, 2, 3, 99};

  for (int i = 0; i < N; i++) {
    if (result[i] != ref[i]) {
      std::cout << "test_6 run failed!\n";
      exit(-1);
    }
  }

  std::cout << "test_6 run passed!\n";
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
