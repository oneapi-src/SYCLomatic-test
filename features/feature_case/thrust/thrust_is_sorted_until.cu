// ====------ thrust_is_sorted_until.cu--------------- *- CUDA -*----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

void test_1() {

  int A[8] = {0, 1, 2, 3, 0, 1, 2, 3};
  int *B = thrust::is_sorted_until(thrust::host, A, A + 8);

  int ref[4] = {0, 1, 2, 3};

  for (int i = 0; i < 4; i++) {
    if (A[i] != ref[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }
  if (B - A != 4) {
    printf("test_1 run failed\n");
    exit(-1);
  }

  printf("test_1 passed!\n");
}

void test_2() {

  int A[8] = {0, 1, 2, 3, 0, 1, 2, 3};
  int *B = thrust::is_sorted_until(A, A + 8);
  int ref[4] = {0, 1, 2, 3};

  for (int i = 0; i < 4; i++) {
    if (A[i] != ref[i]) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }
  if (B - A != 4) {
    printf("test_2 run failed\n");
    exit(-1);
  }

  printf("test_2 passed!\n");
}

void test_3() {

  int A[8] = {3, 2, 1, 0, 3, 2, 1, 0};
  thrust::greater<int> comp;
  int *B = thrust::is_sorted_until(thrust::host, A, A + 8, comp);

  int ref[4] = {3, 2, 1, 0};
  for (int i = 0; i < 4; i++) {
    if (A[i] != ref[i]) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }
  if (B - A != 4) {
    printf("test_3 run failed\n");
    exit(-1);
  }

  printf("test_3 passed!\n");
}

void test_4() {

  int A[8] = {3, 2, 1, 0, 3, 2, 1, 0};
  thrust::greater<int> comp;
  int *B = thrust::is_sorted_until(thrust::host, A, A + 8, comp);

  int ref[4] = {3, 2, 1, 0};
  for (int i = 0; i < 4; i++) {
    if (A[i] != ref[i]) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }
  if (B - A != 4) {
    printf("test_4 run failed\n");
    exit(-1);
  }

  printf("test_4 passed!\n");
}

void test_5() {

  int A[8] = {0, 1, 2, 3, 0, 1, 2, 3};
  thrust::host_vector<int> h_A(A, A + 8);

  thrust::host_vector<int>::iterator end;

  end = thrust::is_sorted_until(thrust::host, h_A.begin(), h_A.end());

  int ref[4] = {0, 1, 2, 3};

  for (int i = 0; i < 4; i++) {
    if (h_A[i] != ref[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }
  if (end - h_A.begin() != 4) {
    printf("test_1 run failed\n");
    exit(-1);
  }

  printf("test_5 passed!\n");
}

void test_6() {

  int A[8] = {0, 1, 2, 3, 0, 1, 2, 3};
  thrust::host_vector<int> h_A(A, A + 8);

  thrust::host_vector<int>::iterator end;

  end = thrust::is_sorted_until(h_A.begin(), h_A.end());

  int ref[4] = {0, 1, 2, 3};

  for (int i = 0; i < 4; i++) {
    if (h_A[i] != ref[i]) {
      printf("test_6 run failed\n");
      exit(-1);
    }
  }
  if (end - h_A.begin() != 4) {
    printf("test_6 run failed\n");
    exit(-1);
  }

  printf("test_6 passed!\n");
}

void test_7() {

  int A[8] = {3, 2, 1, 0, 3, 2, 1, 0};
  thrust::host_vector<int> h_A(A, A + 8);
  thrust::greater<int> comp;
  thrust::host_vector<int>::iterator end;
  end = thrust::is_sorted_until(thrust::host, h_A.begin(), h_A.end(), comp);

  int ref[4] = {3, 2, 1, 0};

  for (int i = 0; i < 4; i++) {
    if (h_A[i] != ref[i]) {
      printf("test_6 run failed\n");
      exit(-1);
    }
  }
  if (end - h_A.begin() != 4) {
    printf("test_6 run failed\n");
    exit(-1);
  }

  printf("test_7 passed!\n");
}

void test_8() {

  int A[8] = {3, 2, 1, 0, 3, 2, 1, 0};
  thrust::host_vector<int> h_A(A, A + 8);
  thrust::greater<int> comp;
  thrust::host_vector<int>::iterator end;
  end = thrust::is_sorted_until(h_A.begin(), h_A.end(), comp);

  int ref[4] = {3, 2, 1, 0};

  for (int i = 0; i < 4; i++) {
    if (h_A[i] != ref[i]) {
      printf("test_8 run failed\n");
      exit(-1);
    }
  }
  if (end - h_A.begin() != 4) {
    printf("test_8 run failed\n");
    exit(-1);
  }

  printf("test_8 passed!\n");
}

void test_9() {

  int A[8] = {0, 1, 2, 3, 0, 1, 2, 3};
  thrust::device_vector<int> d_A(A, A + 8);

  thrust::device_vector<int>::iterator end;

  end = thrust::is_sorted_until(thrust::device, d_A.begin(), d_A.end());

  int ref[4] = {0, 1, 2, 3};

  for (int i = 0; i < 4; i++) {
    if (d_A[i] != ref[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }
  if (end - d_A.begin() != 4) {
    printf("test_1 run failed\n");
    exit(-1);
  }

  printf("test_9 passed!\n");
}

void test_10() {

  int A[8] = {0, 1, 2, 3, 0, 1, 2, 3};
  thrust::device_vector<int> d_A(A, A + 8);

  thrust::device_vector<int>::iterator end;

  end = thrust::is_sorted_until(d_A.begin(), d_A.end());

  int ref[4] = {0, 1, 2, 3};

  for (int i = 0; i < 4; i++) {
    if (d_A[i] != ref[i]) {
      printf("test_10 run failed\n");
      exit(-1);
    }
  }
  if (end - d_A.begin() != 4) {
    printf("test_10 run failed\n");
    exit(-1);
  }

  printf("test_10 passed!\n");
}

void test_11() {

  int A[8] = {3, 2, 1, 0, 3, 2, 1, 0};
  thrust::device_vector<int> d_A(A, A + 8);
  thrust::greater<int> comp;
  thrust::device_vector<int>::iterator end;
  end = thrust::is_sorted_until(thrust::device, d_A.begin(), d_A.end(), comp);

  int ref[4] = {3, 2, 1, 0};

  for (int i = 0; i < 4; i++) {
    if (d_A[i] != ref[i]) {
      printf("test_11 run failed\n");
      exit(-1);
    }
  }
  if (end - d_A.begin() != 4) {
    printf("test_11 run failed\n");
    exit(-1);
  }

  printf("test_11 passed!\n");
}

void test_12() {

  int A[8] = {3, 2, 1, 0, 3, 2, 1, 0};
  thrust::device_vector<int> d_A(A, A + 8);
  thrust::greater<int> comp;
  thrust::device_vector<int>::iterator end;
  end = thrust::is_sorted_until(d_A.begin(), d_A.end(), comp);

  int ref[4] = {3, 2, 1, 0};

  for (int i = 0; i < 4; i++) {
    if (d_A[i] != ref[i]) {
      printf("test_12 run failed\n");
      exit(-1);
    }
  }
  if (end - d_A.begin() != 4) {
    printf("test_12 run failed\n");
    exit(-1);
  }

  printf("test_12 passed!\n");
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