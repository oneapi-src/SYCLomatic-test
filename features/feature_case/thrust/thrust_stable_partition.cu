// ====------ thrust_stable_partition.cu------------------------ *- CUDA -*
// -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>

struct is_even {
  __host__ __device__ bool operator()(const int &x) const {
    return (x % 2) == 0;
  }
};

void test_1() {

  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);
  thrust::stable_partition(thrust::host, data, data + N, is_even());

  int ref[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  for (int i = 0; i < N; i++) {
    if (data[i] != ref[i]) {
      std::cout << "test_1 run failed!"
                << "\n";
      exit(-1);
    }
  }
  std::cout << "test_1 run passed!\n";
}

void test_2() {

  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);

  thrust::host_vector<int> host_data(data, data + N);

  thrust::stable_partition(thrust::host, host_data.begin(),
                           host_data.begin() + N, is_even());

  int ref[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  for (int i = 0; i < N; i++) {
    if (host_data[i] != ref[i]) {
      std::cout << "test_2 run failed!"
                << "\n";
      exit(-1);
    }
  }
  std::cout << "test_2 run passed!\n";
}

void test_3() {

  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);

  thrust::device_vector<int> device_data(data, data + N);

  thrust::stable_partition(thrust::device, device_data.begin(),
                           device_data.begin() + N, is_even());

  int ref[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  for (int i = 0; i < N; i++) {
    if (device_data[i] != ref[i]) {
      std::cout << "test_3 run failed!"
                << "\n";
      exit(-1);
    }
  }
  std::cout << "test_3 run passed!\n";
}

void test_4() {

  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);
  thrust::stable_partition(data, data + N, is_even());

  int ref[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  for (int i = 0; i < N; i++) {
    if (data[i] != ref[i]) {
      std::cout << "test_4 run failed!"
                << "\n";
      exit(-1);
    }
  }
  std::cout << "test_4 run passed!\n";
}

void test_5() {

  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);
  thrust::host_vector<int> host_data(data, data + N);

  thrust::stable_partition(host_data.begin(), host_data.begin() + N, is_even());

  int ref[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  for (int i = 0; i < N; i++) {
    if (host_data[i] != ref[i]) {
      std::cout << "test_5 run failed!"
                << "\n";
      exit(-1);
    }
  }
  std::cout << "test_5 run passed!\n";
}

void test_6() {

  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);

  thrust::device_vector<int> device_data(data, data + N);

  thrust::stable_partition(device_data.begin(), device_data.begin() + N,
                           is_even());

  int ref[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  for (int i = 0; i < N; i++) {
    if (device_data[i] != ref[i]) {
      std::cout << "test_6 run failed!"
                << "\n";
      exit(-1);
    }
  }
  std::cout << "test_6 run passed!\n";
}

void test_7() {

  int data[] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  int S[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);
  thrust::stable_partition(thrust::host, data, data + N, S, is_even());
  int ref[] = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
  for (int i = 0; i < N; i++) {
    if (data[i] != ref[i]) {
      std::cout << "test_7 run failed!"
                << "\n";
      exit(-1);
    }
  }
  std::cout << "test_7 run passed!\n";
}

void test_8() {

  int data[] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  int S[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);

  thrust::host_vector<int> host_data(data, data + N);
  thrust::host_vector<int> host_S(S, S + N);

  thrust::stable_partition(thrust::host, host_data.begin(),
                           host_data.begin() + N, host_S.begin(), is_even());
  int ref[] = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
  for (int i = 0; i < N; i++) {
    if (host_data[i] != ref[i]) {
      std::cout << "test_8 run failed!"
                << "\n";
      exit(-1);
    }
  }
  std::cout << "test_8 run passed!\n";
}

void test_9() {

  int data[] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  int S[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);

  thrust::device_vector<int> device_data(data, data + N);
  thrust::device_vector<int> device_s(S, S + N);

  thrust::stable_partition(thrust::device, device_data.begin(),
                           device_data.begin() + N, device_s.begin(),
                           is_even());
  int ref[] = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
  for (int i = 0; i < N; i++) {
    if (device_data[i] != ref[i]) {
      std::cout << "test_9 run failed!"
                << "\n";
      exit(-1);
    }
  }
  std::cout << "test_9 run passed!\n";
}

void test_10() {

  int data[] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  int S[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);
  thrust::stable_partition(data, data + N, S, is_even());

  int ref[] = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
  for (int i = 0; i < N; i++) {
    if (data[i] != ref[i]) {
      std::cout << "test_10 run failed!"
                << "\n";
      exit(-1);
    }
  }
  std::cout << "test_10 run passed!\n";
}

void test_11() {

  int data[] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  int S[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);

  thrust::host_vector<int> host_data(data, data + N);
  thrust::host_vector<int> host_S(S, S + N);

  thrust::stable_partition(host_data.begin(), host_data.begin() + N,
                           host_S.begin(), is_even());
  int ref[] = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
  for (int i = 0; i < N; i++) {
    if (host_data[i] != ref[i]) {
      std::cout << "test_11 run failed!"
                << "\n";
      exit(-1);
    }
  }
  std::cout << "test_11 run passed!\n";
}

void test_12() {

  int data[] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  int S[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);

  thrust::device_vector<int> device_data(data, data + N);
  thrust::device_vector<int> device_s(S, S + N);

  thrust::stable_partition(device_data.begin(), device_data.begin() + N,
                           device_s.begin(), is_even());
  int ref[] = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
  for (int i = 0; i < N; i++) {
    if (device_data[i] != ref[i]) {
      std::cout << "test_12 run failed!"
                << "\n";
      exit(-1);
    }
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
