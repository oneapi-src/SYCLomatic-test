// ====------ partition.cu--------------- *- CUDA -* -------------------===//
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
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

struct is_even {
  __host__ __device__ bool operator()(const int &x) const {
    return (x % 2) == 0;
  }
};

void test_1() { // host iterator

  int datas[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int ans[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  const int N = sizeof(datas) / sizeof(int);
  thrust::host_vector<int> v(datas, datas + N);

  thrust::partition(thrust::host, v.begin(), v.end(), is_even());
  for (int i = 0; i < N / 2; i++) {
    int j = 0;
    for (; j < N / 2; j++) {
      if (v[i] == ans[j]) {
        break;
      }
    }
    if (j >= N / 2) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }

  for (int i = N / 2; i < N; i++) {
    int j = N / 2;
    for (; j < N; j++) {
      if (v[i] == ans[j])
        break;
    }
    if (j >= N) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }

  printf("test_1 run passed!\n");
}

void test_2() { // host iterator

  int datas[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int ans[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  const int N = sizeof(datas) / sizeof(int);
  thrust::host_vector<int> v(datas, datas + N);

  thrust::partition(v.begin(), v.end(), is_even());
  for (int i = 0; i < N / 2; i++) {
    int j = 0;
    for (; j < N / 2; j++) {
      if (v[i] == ans[j]) {
        break;
      }
    }
    if (j >= N / 2) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }

  for (int i = N / 2; i < N; i++) {
    int j = N / 2;
    for (; j < N; j++) {
      if (v[i] == ans[j])
        break;
    }
    if (j >= N) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }

  printf("test_2 run passed!\n");
}

void test_3() { // host iterator

  const int N = 10;
  int datas[N] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  int ans[N] = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
  int stencil[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  thrust::host_vector<int> vdata(datas, datas + N);
  thrust::host_vector<int> vstencil(stencil, stencil + N);

  thrust::partition(thrust::host, vdata.begin(), vdata.end(), vstencil.begin(),
                    is_even());
  for (int i = 0; i < N / 2; i++) {
    int j = 0;
    for (; j < N / 2; j++) {
      if (vdata[i] == ans[j]) {
        break;
      }
    }
    if (j >= N / 2) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }

  for (int i = N / 2; i < N; i++) {
    int j = N / 2;
    for (; j < N; j++) {
      if (vdata[i] == ans[j])
        break;
    }
    if (j >= N) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }

  printf("test_3 run passed!\n");
}

void test_4() { // host iterator

  const int N = 10;
  int datas[N] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  int ans[N] = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
  int stencil[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  thrust::host_vector<int> vdata(datas, datas + N);
  thrust::host_vector<int> vstencil(stencil, stencil + N);

  thrust::partition(vdata.begin(), vdata.end(), vstencil.begin(), is_even());
  for (int i = 0; i < N / 2; i++) {
    int j = 0;
    for (; j < N / 2; j++) {
      if (vdata[i] == ans[j]) {
        break;
      }
    }
    if (j >= N / 2) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }

  for (int i = N / 2; i < N; i++) {
    int j = N / 2;
    for (; j < N; j++) {
      if (vdata[i] == ans[j])
        break;
    }
    if (j >= N) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }

  printf("test_4 run passed!\n");
}

void test_5() { // device iterator

  int datas[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int ans[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  const int N = sizeof(datas) / sizeof(int);
  thrust::device_vector<int> v(datas, datas + N);

  thrust::partition(thrust::device, v.begin(), v.end(), is_even());
  for (int i = 0; i < N / 2; i++) {
    int j = 0;
    for (; j < N / 2; j++) {
      if (v[i] == ans[j]) {
        break;
      }
    }
    if (j >= N / 2) {
      printf("test_5 run failed\n");
      exit(-1);
    }
  }

  for (int i = N / 2; i < N; i++) {
    int j = N / 2;
    for (; j < N; j++) {
      if (v[i] == ans[j])
        break;
    }
    if (j >= N) {
      printf("test_5 run failed\n");
      exit(-1);
    }
  }

  printf("test_5 run passed!\n");
}

void test_6() { // device iterator

  int datas[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int ans[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  const int N = sizeof(datas) / sizeof(int);
  thrust::device_vector<int> v(datas, datas + N);

  thrust::partition(v.begin(), v.end(), is_even());
  for (int i = 0; i < N / 2; i++) {
    int j = 0;
    for (; j < N / 2; j++) {
      if (v[i] == ans[j]) {
        break;
      }
    }
    if (j >= N / 2) {
      printf("test_6 run failed\n");
      exit(-1);
    }
  }

  for (int i = N / 2; i < N; i++) {
    int j = N / 2;
    for (; j < N; j++) {
      if (v[i] == ans[j])
        break;
    }
    if (j >= N) {
      printf("test_6 run failed\n");
      exit(-1);
    }
  }

  printf("test_6 run passed!\n");
}

void test_7() { // device iterator

  const int N = 10;
  int datas[N] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  int ans[N] = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
  int stencil[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  thrust::device_vector<int> vdata(datas, datas + N);
  thrust::device_vector<int> vstencil(stencil, stencil + N);

  thrust::partition(thrust::device, vdata.begin(), vdata.end(),
                    vstencil.begin(), is_even());
  for (int i = 0; i < N / 2; i++) {
    int j = 0;
    for (; j < N / 2; j++) {
      if (vdata[i] == ans[j]) {
        break;
      }
    }
    if (j >= N / 2) {
      printf("test_7 run failed\n");
      exit(-1);
    }
  }

  for (int i = N / 2; i < N; i++) {
    int j = N / 2;
    for (; j < N; j++) {
      if (vdata[i] == ans[j])
        break;
    }
    if (j >= N) {
      printf("test_7 run failed\n");
      exit(-1);
    }
  }

  printf("test_7 run passed!\n");
}

void test_8() { // device iterator

  const int N = 10;
  int datas[N] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  int ans[N] = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
  int stencil[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  thrust::device_vector<int> vdata(datas, datas + N);
  thrust::device_vector<int> vstencil(stencil, stencil + N);

  thrust::partition(vdata.begin(), vdata.end(), vstencil.begin(), is_even());
  for (int i = 0; i < N / 2; i++) {
    int j = 0;
    for (; j < N / 2; j++) {
      if (vdata[i] == ans[j]) {
        break;
      }
    }
    if (j >= N / 2) {
      printf("test_8 run failed\n");
      exit(-1);
    }
  }

  for (int i = N / 2; i < N; i++) {
    int j = N / 2;
    for (; j < N; j++) {
      if (vdata[i] == ans[j])
        break;
    }
    if (j >= N) {
      printf("test_8 run failed\n");
      exit(-1);
    }
  }

  printf("test_8 run passed!\n");
}

void test_9() { // raw ptr

  int datas[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int ans[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  const int N = sizeof(datas) / sizeof(int);

  thrust::partition(thrust::host, datas, datas + N, is_even());
  for (int i = 0; i < N / 2; i++) {
    int j = 0;
    for (; j < N / 2; j++) {
      if (datas[i] == ans[j]) {
        break;
      }
    }
    if (j >= N / 2) {
      printf("test_9 run failed\n");
      exit(-1);
    }
  }

  for (int i = N / 2; i < N; i++) {
    int j = N / 2;
    for (; j < N; j++) {
      if (datas[i] == ans[j])
        break;
    }
    if (j >= N) {
      printf("test_9 run failed\n");
      exit(-1);
    }
  }

  printf("test_9 run passed!\n");
}

void test_10() { // raw ptr

  int datas[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int ans[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  const int N = sizeof(datas) / sizeof(int);

  thrust::partition(datas, datas + N, is_even());
  for (int i = 0; i < N / 2; i++) {
    int j = 0;
    for (; j < N / 2; j++) {
      if (datas[i] == ans[j]) {
        break;
      }
    }
    if (j >= N / 2) {
      printf("test_10 run failed\n");
      exit(-1);
    }
  }

  for (int i = N / 2; i < N; i++) {
    int j = N / 2;
    for (; j < N; j++) {
      if (datas[i] == ans[j])
        break;
    }
    if (j >= N) {
      printf("test_10 run failed\n");
      exit(-1);
    }
  }

  printf("test_10 run passed!\n");
}

void test_11() { // raw ptr

  const int N = 10;
  int datas[N] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  int ans[N] = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
  int stencil[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  thrust::partition(thrust::host, datas, datas + N, stencil, is_even());
  for (int i = 0; i < N / 2; i++) {
    int j = 0;
    for (; j < N / 2; j++) {
      if (datas[i] == ans[j]) {
        break;
      }
    }
    if (j >= N / 2) {
      printf("test_11 run failed\n");
      exit(-1);
    }
  }

  for (int i = N / 2; i < N; i++) {
    int j = N / 2;
    for (; j < N; j++) {
      if (datas[i] == ans[j])
        break;
    }
    if (j >= N) {
      printf("test_11 run failed\n");
      exit(-1);
    }
  }

  printf("test_11 run passed!\n");
}

void test_12() { // raw ptr

  const int N = 10;
  int datas[N] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  int ans[N] = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
  int stencil[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  thrust::partition(datas, datas + N, stencil, is_even());
  for (int i = 0; i < N / 2; i++) {
    int j = 0;
    for (; j < N / 2; j++) {
      if (datas[i] == ans[j]) {
        break;
      }
    }
    if (j >= N / 2) {
      printf("test_12 run failed\n");
      exit(-1);
    }
  }

  for (int i = N / 2; i < N; i++) {
    int j = N / 2;
    for (; j < N; j++) {
      if (datas[i] == ans[j])
        break;
    }
    if (j >= N) {
      printf("test_12 run failed\n");
      exit(-1);
    }
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
