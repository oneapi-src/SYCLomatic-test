// ====------ thrust_remove.cu------------------ *- CUDA -*
// -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===--------------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>

void test_1() {

  const int N = 6;
  int data[N] = {3, 1, 4, 1, 5, 9};
  int *new_end = thrust::remove(thrust::host, data, data + N, 1);

  int Size = new_end - data;
  if (Size != 4) {
    std::cout << "test_1 run failed!\n";
  }

  int ref[] = {3, 4, 5, 9};
  for (int i = 0; i < Size; i++) {
    if (data[i] != ref[i]) {
      std::cout << "test_1 run failed!\n";
      exit(-1);
    }
  }
  std::cout << "test_1 run passed!\n";
}

void test_2() {

  const int N = 6;
  int data[N] = {3, 1, 4, 1, 5, 9};

  thrust::host_vector<int> host_data(data, data + N);

  typename thrust::host_vector<int>::iterator new_end =
      thrust::remove(thrust::host, host_data.begin(), host_data.begin() + N, 1);

  int Size = new_end - host_data.begin();
  if (Size != 4) {
    std::cout << "test_1 run failed!\n";
  }

  int ref[] = {3, 4, 5, 9};
  for (int i = 0; i < Size; i++) {
    if (host_data[i] != ref[i]) {
      std::cout << "test_2 run failed!\n";
      exit(-1);
    }
  }
  std::cout << "test_2 run passed!\n";
}

void test_3() {

  const int N = 6;
  int data[N] = {3, 1, 4, 1, 5, 9};

  thrust::device_vector<int> device_data(data, data + N);

  typename thrust::device_vector<int>::iterator new_end = thrust::remove(
      thrust::device, device_data.begin(), device_data.begin() + N, 1);

  int Size = new_end - device_data.begin();

  if (Size != 4) {
    std::cout << "test_1 run failed!\n";
  }

  int ref[] = {3, 4, 5, 9};
  for (int i = 0; i < Size; i++) {
    if (device_data[i] != ref[i]) {
      std::cout << "test_3 run failed!\n";
      exit(-1);
    }
  }
  std::cout << "test_3 run passed!\n";
}

void test_4() {

  const int N = 6;
  int data[N] = {3, 1, 4, 1, 5, 9};
  int *new_end = thrust::remove(data, data + N, 1);

  int Size = new_end - data;

  if (Size != 4) {
    std::cout << "test_1 run failed!\n";
  }

  int ref[] = {3, 4, 5, 9};
  for (int i = 0; i < Size; i++) {
    if (data[i] != ref[i]) {
      std::cout << "test_4 run failed!\n";
      exit(-1);
    }
  }
  std::cout << "test_4 run passed!\n";
}

void test_5() {

  const int N = 6;
  int data[N] = {3, 1, 4, 1, 5, 9};

  thrust::host_vector<int> host_data(data, data + N);

  typename thrust::host_vector<int>::iterator new_end =
      thrust::remove(host_data.begin(), host_data.begin() + N, 1);

  int Size = new_end - host_data.begin();

  if (Size != 4) {
    std::cout << "test_1 run failed!\n";
  }

  int ref[] = {3, 4, 5, 9};
  for (int i = 0; i < Size; i++) {
    if (host_data[i] != ref[i]) {
      std::cout << "test_5 run failed!\n";
      exit(-1);
    }
  }
  std::cout << "test_5 run passed!\n";
}

void test_6() {

  const int N = 6;
  int data[N] = {3, 1, 4, 1, 5, 9};

  thrust::device_vector<int> device_data(data, data + N);

  typename thrust::device_vector<int>::iterator new_end =
      thrust::remove(device_data.begin(), device_data.begin() + N, 1);

  int Size = new_end - device_data.begin();

  if (Size != 4) {
    std::cout << "test_1 run failed!\n";
  }

  int ref[] = {3, 4, 5, 9};
  for (int i = 0; i < Size; i++) {
    if (device_data[i] != ref[i]) {
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
