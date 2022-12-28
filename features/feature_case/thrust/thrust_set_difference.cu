// ====------ thrust_set_different.cu--------------- *- CUDA -* -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===------------------------------------------------------------------------------===//

#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/set_operations.h>

void test_1() {
  const int N = 7, M = 5, P = 3;
  int A[N] = {0, 1, 3, 4, 5, 6, 9};
  int B[M] = {1, 3, 5, 7, 9};
  int C[P];
  int ans[P] = {0, 4, 6};
  thrust::host_vector<int> VA(A, A + N);
  thrust::host_vector<int> VB(B, B + M);
  thrust::host_vector<int> VC(C, C + P);

  thrust::host_vector<int>::iterator result_end = thrust::set_difference(
      thrust::host, VA.begin(), VA.end(), VB.begin(), VB.end(), VC.begin());
  for (int i = 0; i < P; i++) {
    if (VC[i] != ans[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }

  printf("test_1 run passed!\n");
}

void test_2() {

  const int N = 7, M = 5, P = 3;
  int A[N] = {0, 1, 3, 4, 5, 6, 9};
  int B[M] = {1, 3, 5, 7, 9};
  int C[P];
  int ans[P] = {0, 4, 6};
  thrust::host_vector<int> VA(A, A + N);
  thrust::host_vector<int> VB(B, B + M);
  thrust::host_vector<int> VC(C, C + P);

  thrust::host_vector<int>::iterator result_end = thrust::set_difference(
      VA.begin(), VA.end(), VB.begin(), VB.end(), VC.begin());
  for (int i = 0; i < P; i++) {
    if (VC[i] != ans[i]) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }

  printf("test_2 run passed!\n");
}

void test_3() {
  const int N = 7, M = 5, P = 3;
  int A[N] = {9, 6, 5, 4, 3, 1, 0};
  int B[M] = {9, 7, 5, 3, 1};
  int C[P];
  int ans[P] = {6, 4, 0};
  thrust::host_vector<int> VA(A, A + N);
  thrust::host_vector<int> VB(B, B + M);
  thrust::host_vector<int> VC(C, C + P);

  thrust::host_vector<int>::iterator result_end =
      thrust::set_difference(thrust::host, VA.begin(), VA.end(), VB.begin(),
                             VB.end(), VC.begin(), thrust::greater<int>());
  for (int i = 0; i < P; i++) {
    if (VC[i] != ans[i]) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }

  printf("test_3 run passed!\n");
}

void test_4() {

  const int N = 7, M = 5, P = 3;
  int A[N] = {9, 6, 5, 4, 3, 1, 0};
  int B[M] = {9, 7, 5, 3, 1};
  int C[P];
  int ans[P] = {6, 4, 0};
  thrust::host_vector<int> VA(A, A + N);
  thrust::host_vector<int> VB(B, B + M);
  thrust::host_vector<int> VC(C, C + P);

  thrust::host_vector<int>::iterator result_end =
      thrust::set_difference(VA.begin(), VA.end(), VB.begin(), VB.end(),
                             VC.begin(), thrust::greater<int>());
  for (int i = 0; i < P; i++) {
    if (VC[i] != ans[i]) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }

  printf("test_4 run passed!\n");
}

void test_5() {
  const int N = 7, M = 5, P = 3;
  int A[N] = {0, 1, 3, 4, 5, 6, 9};
  int B[M] = {1, 3, 5, 7, 9};
  int C[P];
  int ans[P] = {0, 4, 6};
  thrust::device_vector<int> VA(A, A + N);
  thrust::device_vector<int> VB(B, B + M);
  thrust::device_vector<int> VC(C, C + P);

  thrust::device_vector<int>::iterator result_end = thrust::set_difference(
      thrust::device, VA.begin(), VA.end(), VB.begin(), VB.end(), VC.begin());
  for (int i = 0; i < P; i++) {
    if (VC[i] != ans[i]) {
      printf("test_5 run failed\n");
      exit(-1);
    }
  }

  printf("test_5 run passed!\n");
}

void test_6() {

  const int N = 7, M = 5, P = 3;
  int A[N] = {0, 1, 3, 4, 5, 6, 9};
  int B[M] = {1, 3, 5, 7, 9};
  int C[P];
  int ans[P] = {0, 4, 6};
  thrust::device_vector<int> VA(A, A + N);
  thrust::device_vector<int> VB(B, B + M);
  thrust::device_vector<int> VC(C, C + P);

  thrust::device_vector<int>::iterator result_end = thrust::set_difference(
      VA.begin(), VA.end(), VB.begin(), VB.end(), VC.begin());
  for (int i = 0; i < P; i++) {
    if (VC[i] != ans[i]) {
      printf("test_6 run failed\n");
      exit(-1);
    }
  }

  printf("test_6 run passed!\n");
}

void test_7() {
  const int N = 7, M = 5, P = 3;
  int A[N] = {9, 6, 5, 4, 3, 1, 0};
  int B[M] = {9, 7, 5, 3, 1};
  int C[P];
  int ans[P] = {6, 4, 0};
  thrust::device_vector<int> VA(A, A + N);
  thrust::device_vector<int> VB(B, B + M);
  thrust::device_vector<int> VC(C, C + P);

  thrust::device_vector<int>::iterator result_end =
      thrust::set_difference(thrust::device, VA.begin(), VA.end(), VB.begin(),
                             VB.end(), VC.begin(), thrust::greater<int>());
  for (int i = 0; i < P; i++) {
    if (VC[i] != ans[i]) {
      printf("test_7 run failed\n");
      exit(-1);
    }
  }

  printf("test_7 run passed!\n");
}

void test_8() {

  const int N = 7, M = 5, P = 3;
  int A[N] = {9, 6, 5, 4, 3, 1, 0};
  int B[M] = {9, 7, 5, 3, 1};
  int C[P];
  int ans[P] = {6, 4, 0};
  thrust::device_vector<int> VA(A, A + N);
  thrust::device_vector<int> VB(B, B + M);
  thrust::device_vector<int> VC(C, C + P);

  thrust::device_vector<int>::iterator result_end =
      thrust::set_difference(VA.begin(), VA.end(), VB.begin(), VB.end(),
                             VC.begin(), thrust::greater<int>());
  for (int i = 0; i < P; i++) {
    if (VC[i] != ans[i]) {
      printf("test_8 run failed\n");
      exit(-1);
    }
  }

  printf("test_8 run passed!\n");
}

void test_9() {
  const int N = 7, M = 5, P = 3;
  int A[N] = {0, 1, 3, 4, 5, 6, 9};
  int B[M] = {1, 3, 5, 7, 9};
  int C[P];
  int ans[P] = {0, 4, 6};

  thrust::set_difference(thrust::host, A, A + N, B, B + M, C);
  for (int i = 0; i < P; i++) {
    if (C[i] != ans[i]) {
      printf("test_9 run failed\n");
      exit(-1);
    }
  }

  printf("test_9 run passed!\n");
}

void test_10() {

  const int N = 7, M = 5, P = 3;
  int A[N] = {0, 1, 3, 4, 5, 6, 9};
  int B[M] = {1, 3, 5, 7, 9};
  int C[P];
  int ans[P] = {0, 4, 6};

  thrust::set_difference(A, A + N, B, B + M, C);
  for (int i = 0; i < P; i++) {
    if (C[i] != ans[i]) {
      printf("test_10 run failed\n");
      exit(-1);
    }
  }

  printf("test_10 run passed!\n");
}

void test_11() {
  const int N = 7, M = 5, P = 3;
  int A[N] = {9, 6, 5, 4, 3, 1, 0};
  int B[M] = {9, 7, 5, 3, 1};
  int C[P];
  int ans[P] = {6, 4, 0};
  thrust::set_difference(thrust::host, A, A + N, B, B + M, C,
                         thrust::greater<int>());
  for (int i = 0; i < P; i++) {
    if (C[i] != ans[i]) {
      printf("test_11 run failed\n");
      exit(-1);
    }
  }

  printf("test_11 run passed!\n");
}

void test_12() {

  const int N = 7, M = 5, P = 3;
  int A[N] = {9, 6, 5, 4, 3, 1, 0};
  int B[M] = {9, 7, 5, 3, 1};
  int C[P];
  int ans[P] = {6, 4, 0};
  thrust::set_difference(A, A + N, B, B + M, C, thrust::greater<int>());
  for (int i = 0; i < P; i++) {
    if (C[i] != ans[i]) {
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