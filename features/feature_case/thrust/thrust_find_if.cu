// ====------ thrust_find_if.cu---------- *- CUDA -* -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===-------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/host_vector.h>

struct greater_than_four {
  __host__ __device__ bool operator()(int x) const { return x > 4; }
};

void test_1() {

  int data[4] = {0, 5, 3, 7};

  int *ptr = thrust::find_if(data, data + 3, greater_than_four());

  if (*ptr != 5) {
    std::cout << "test_1 run failed!\n";
    exit(-1);
  }

  std::cout << "test_1 run passed!\n";
}

void test_2() {
  const int N = 4;
  int data[4] = {0, 5, 3, 7};
  thrust::device_vector<int> device_data(data, data + N);

  thrust::device_vector<int>::iterator iter;
  iter = thrust::find_if(device_data.begin(), device_data.end(),
                         greater_than_four());

  if (*iter != 5) {
    std::cout << "test_2 run failed!\n";
    exit(-1);
  }

  std::cout << "test_2 run passed!\n";
}

void test_3() {
  const int N = 4;
  int data[4] = {0, 5, 3, 7};
  thrust::host_vector<int> host_data(data, data + N);

  thrust::host_vector<int>::iterator iter;
  iter =
      thrust::find_if(host_data.begin(), host_data.end(), greater_than_four());

  if (*iter != 5) {
    std::cout << "test_3 run failed!\n";
    exit(-1);
  }

  std::cout << "test_3 run passed!\n";
}

void test_4() {

  int data[4] = {0, 5, 3, 7};

  int *ptr = thrust::find_if(thrust::host, data, data + 3, greater_than_four());

  if (*ptr != 5) {
    std::cout << "test_4 run failed!\n";
    exit(-1);
  }

  std::cout << "test_4 run passed!\n";
}

void test_5() {
  const int N = 4;
  int data[4] = {0, 5, 3, 7};
  thrust::device_vector<int> device_data(data, data + N);

  thrust::device_vector<int>::iterator iter;
  iter = thrust::find_if(thrust::device, device_data.begin(), device_data.end(),
                         greater_than_four());

  if (*iter != 5) {
    std::cout << "test_5 run failed!\n";
    exit(-1);
  }

  std::cout << "test_5 run passed!\n";
}

void test_6() {
  const int N = 4;
  int data[4] = {0, 5, 3, 7};
  thrust::host_vector<int> host_data(data, data + N);

  thrust::host_vector<int>::iterator iter;
  iter = thrust::find_if(thrust::host, host_data.begin(), host_data.end(),
                         greater_than_four());

  if (*iter != 5) {
    std::cout << "test_6 run failed!\n";
    exit(-1);
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
