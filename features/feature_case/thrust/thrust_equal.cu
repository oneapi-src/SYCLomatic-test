// ====------ thrust_equal.cu------------- *- CUDA -*---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------- ---------------===//

#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

void test_1() {

  const int N = 7;

  int A1[N] = {3, 1, 4, 1, 5, 9, 3};
  int A2[N] = {3, 1, 4, 2, 8, 5, 7};
  bool result = thrust::equal(thrust::host, A1, A1 + N, A2);

  if (result) {
    printf("test_1 run failed\n");
    exit(-1);
  }

  printf("test_1 run passed!\n");
}

void test_2() {

  const int N = 7;
  int A1[N] = {3, 1, 4, 1, 5, 9, 3};
  int A2[N] = {3, 1, 4, 2, 8, 5, 7};
  bool result = thrust::equal(A1, A1 + N, A2);

  if (result) {
    printf("test_2 run failed\n");
    exit(-1);
  }

  printf("test_2 run passed!\n");
}

struct compare_modulo_two {
  __host__ __device__ bool operator()(int x, int y) const {
    return (x % 2) == (y % 2);
  }
};

void test_3() {

  const int N = 6;

  int x[N] = {0, 2, 4, 6, 8, 10};
  int y[N] = {1, 3, 5, 7, 9, 11};
  bool result = thrust::equal(x, x + N, y, compare_modulo_two());

  if (result) {
    printf("test_3 run failed\n");
    exit(-1);
  }

  printf("test_3 run passed!\n");
}

void test_4() {

  const int N = 6;

  int x[N] = {0, 2, 4, 6, 8, 10};
  int y[N] = {1, 3, 5, 7, 9, 11};
  bool result = thrust::equal(thrust::host, x, x + N, y, compare_modulo_two());

  if (result) {
    printf("test_4 run failed\n");
    exit(-1);
  }

  printf("test_4 run passed!\n");
}

void test_5() {

  const int N = 7;

  int A1[N] = {3, 1, 4, 1, 5, 9, 3};
  int A2[N] = {3, 1, 4, 2, 8, 5, 7};

  thrust::host_vector<int> h_A1(A1, A1 + N);
  thrust::host_vector<int> h_A2(A2, A2 + N);

  bool result =
      thrust::equal(thrust::host, h_A1.begin(), h_A1.end(), h_A2.begin());

  if (result) {
    printf("test_5 run failed\n");
    exit(-1);
  }

  printf("test_5 run passed!\n");
}

void test_6() {
  const int N = 7;
  int A1[N] = {3, 1, 4, 1, 5, 9, 3};
  int A2[N] = {3, 1, 4, 2, 8, 5, 7};

  thrust::host_vector<int> h_A1(A1, A1 + N);
  thrust::host_vector<int> h_A2(A2, A2 + N);

  bool result = thrust::equal(h_A1.begin(), h_A1.end(), h_A2.begin());

  if (result) {
    printf("test_6 run failed\n");
    exit(-1);
  }

  printf("test_6 run passed!\n");
}

void test_7() {
  const int N = 6;
  int x[N] = {0, 2, 4, 6, 8, 10};
  int y[N] = {1, 3, 5, 7, 9, 11};

  thrust::host_vector<int> h_x(x, x + N);
  thrust::host_vector<int> h_y(y, y + N);

  bool result =
      thrust::equal(h_x.begin(), h_x.end(), h_y.begin(), compare_modulo_two());

  if (result) {
    printf("test_7 run failed\n");
    exit(-1);
  }

  printf("test_7 run passed!\n");
}

void test_8() {
  const int N = 6;
  int x[N] = {0, 2, 4, 6, 8, 10};
  int y[N] = {1, 3, 5, 7, 9, 11};

  thrust::host_vector<int> h_x(x, x + N);
  thrust::host_vector<int> h_y(y, y + N);
  bool result = thrust::equal(thrust::host, h_x.begin(), h_x.end(), h_y.begin(),
                              compare_modulo_two());

  if (result) {
    printf("test_8 run failed\n");
    exit(-1);
  }

  printf("test_8 run passed!\n");
}

void test_9() {

  const int N = 7;

  int A1[N] = {3, 1, 4, 1, 5, 9, 3};
  int A2[N] = {3, 1, 4, 2, 8, 5, 7};

  thrust::device_vector<int> d_A1(A1, A1 + N);
  thrust::device_vector<int> d_A2(A2, A2 + N);

  bool result =
      thrust::equal(thrust::device, d_A1.begin(), d_A1.end(), d_A2.begin());

  if (result) {
    printf("test_9 run failed\n");
    exit(-1);
  }

  printf("test_9 run passed!\n");
}

void test_10() {
  const int N = 7;
  int A1[N] = {3, 1, 4, 1, 5, 9, 3};
  int A2[N] = {3, 1, 4, 2, 8, 5, 7};

  thrust::device_vector<int> d_A1(A1, A1 + N);
  thrust::device_vector<int> d_A2(A2, A2 + N);

  bool result = thrust::equal(d_A1.begin(), d_A1.end(), d_A2.begin());

  if (result) {
    printf("test_10 run failed\n");
    exit(-1);
  }

  printf("test_10 run passed!\n");
}

void test_11() {
  const int N = 6;
  int x[N] = {0, 2, 4, 6, 8, 10};
  int y[N] = {1, 3, 5, 7, 9, 11};

  thrust::device_vector<int> d_x(x, x + N);
  thrust::device_vector<int> d_y(y, y + N);

  bool result =
      thrust::equal(d_x.begin(), d_x.end(), d_y.begin(), compare_modulo_two());

  if (result) {
    printf("test_11 run failed\n");
    exit(-1);
  }

  printf("test_11 run passed!\n");
}

void test_12() {
  const int N = 6;
  int x[N] = {0, 2, 4, 6, 8, 10};
  int y[N] = {1, 3, 5, 7, 9, 11};

  thrust::device_vector<int> d_x(x, x + N);
  thrust::device_vector<int> d_y(y, y + N);
  bool result = thrust::equal(thrust::device, d_x.begin(), d_x.end(),
                              d_y.begin(), compare_modulo_two());

  if (result) {
    printf("test_12 run failed\n");
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