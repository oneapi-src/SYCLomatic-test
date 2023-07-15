// ====------ thrust_set_union.cu------------- *- CUDA -* ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/set_operations.h>

void test_1() {
  int A1[7] = {0, 2, 4, 6, 8, 10, 12};
  int A2[5] = {1, 3, 5, 7, 9};
  int result[12];
  int *result_end =
      thrust::set_union(thrust::host, A1, A1 + 7, A2, A2 + 5, result);

  int ref[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12};
  for (int i = 0; i < 8; i++) {
    if (result[i] != ref[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }

  printf("test_1 run pass\n");
}

void test_2() {
  int A1[7] = {0, 2, 4, 6, 8, 10, 12};
  int A2[5] = {1, 3, 5, 7, 9};
  int result[12];
  int *result_end = thrust::set_union(A1, A1 + 7, A2, A2 + 5, result);

  int ref[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12};
  for (int i = 0; i < 8; i++) {
    if (result[i] != ref[i]) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }

  printf("test_2 run pass\n");
}

void test_3() {
  int A1[7] = {12, 10, 8, 6, 4, 2, 0};
  int A2[5] = {9, 7, 5, 3, 1};
  int result[12];
  int *result_end = thrust::set_union(thrust::host, A1, A1 + 7, A2, A2 + 5,
                                      result, thrust::greater<int>());

  int ref[12] = {12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  for (int i = 0; i < 8; i++) {
    if (result[i] != ref[i]) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }

  printf("test_3 run pass\n");
}

void test_4() {
  int A1[7] = {12, 10, 8, 6, 4, 2, 0};
  int A2[5] = {9, 7, 5, 3, 1};
  int result[11];
  int *result_end =
      thrust::set_union(A1, A1 + 7, A2, A2 + 5, result, thrust::greater<int>());

  int ref[12] = {12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  for (int i = 0; i < 8; i++) {
    if (result[i] != ref[i]) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }

  printf("test_4 run pass\n");
}

void test_5() {
  int A1[7] = {0, 2, 4, 6, 8, 10, 12};
  int A2[5] = {1, 3, 5, 7, 9};

  thrust::device_vector<int> d_A1(A1, A1 + 7);
  thrust::device_vector<int> d_A2(A2, A2 + 5);
  thrust::device_vector<int> d_result(11);
  typedef thrust::device_vector<int>::iterator Iterator;

  Iterator result_end =
      thrust::set_union(thrust::device, d_A1.begin(), d_A1.end(), d_A2.begin(),
                        d_A2.end(), d_result.begin());

  int ref[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12};
  for (int i = 0; i < 8; i++) {
    if (d_result[i] != ref[i]) {
      printf("test_5 run failed\n");
      exit(-1);
    }
  }

  printf("test_5 run pass\n");
}

void test_6() {
  int A1[7] = {0, 2, 4, 6, 8, 10, 12};
  int A2[5] = {1, 3, 5, 7, 9};

  thrust::device_vector<int> d_A1(A1, A1 + 7);
  thrust::device_vector<int> d_A2(A2, A2 + 5);
  thrust::device_vector<int> d_result(11);
  typedef thrust::device_vector<int>::iterator Iterator;

  Iterator result_end = thrust::set_union(
      d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());

  int ref[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12};
  for (int i = 0; i < 8; i++) {
    if (d_result[i] != ref[i]) {
      printf("test_6 run failed\n");
      exit(-1);
    }
  }

  printf("test_6 run pass\n");
}

void test_7() {
  int A1[7] = {12, 10, 8, 6, 4, 2, 0};
  int A2[5] = {9, 7, 5, 3, 1};

  thrust::device_vector<int> d_A1(A1, A1 + 7);
  thrust::device_vector<int> d_A2(A2, A2 + 5);
  thrust::device_vector<int> d_result(12);
  typedef thrust::device_vector<int>::iterator Iterator;

  Iterator result_end =
      thrust::set_union(thrust::device, d_A1.begin(), d_A1.end(), d_A2.begin(),
                        d_A2.end(), d_result.begin(), thrust::greater<int>());

  int ref[12] = {12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  for (int i = 0; i < 8; i++) {
    if (d_result[i] != ref[i]) {
      printf("test_7 run failed\n");
      exit(-1);
    }
  }

  printf("test_7 run pass\n");
}

void test_8() {
  int A1[7] = {12, 10, 8, 6, 4, 2, 0};
  int A2[5] = {9, 7, 5, 3, 1};

  thrust::device_vector<int> d_A1(A1, A1 + 7);
  thrust::device_vector<int> d_A2(A2, A2 + 5);
  thrust::device_vector<int> d_result(12);
  typedef thrust::device_vector<int>::iterator Iterator;

  Iterator result_end =
      thrust::set_union(d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(),
                        d_result.begin(), thrust::greater<int>());

  int ref[12] = {12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  for (int i = 0; i < 8; i++) {
    if (d_result[i] != ref[i]) {
      printf("test_8 run failed\n");
      exit(-1);
    }
  }

  printf("test_8 run pass\n");
}

void test_9() {
  int A1[7] = {0, 2, 4, 6, 8, 10, 12};
  int A2[5] = {1, 3, 5, 7, 9};

  thrust::host_vector<int> h_A1(A1, A1 + 7);
  thrust::host_vector<int> h_A2(A2, A2 + 5);
  thrust::host_vector<int> h_result(12);
  typedef thrust::host_vector<int>::iterator Iterator;

  Iterator result_end =
      thrust::set_union(thrust::host, h_A1.begin(), h_A1.end(), h_A2.begin(),
                        h_A2.end(), h_result.begin());

  int ref[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12};
  for (int i = 0; i < 8; i++) {
    if (h_result[i] != ref[i]) {
      printf("test_9 run failed\n");
      exit(-1);
    }
  }

  printf("test_9 run pass\n");
}

void test_10() {
  int A1[7] = {0, 2, 4, 6, 8, 10, 12};
  int A2[5] = {1, 3, 5, 7, 9};

  thrust::host_vector<int> h_A1(A1, A1 + 7);
  thrust::host_vector<int> h_A2(A2, A2 + 5);
  thrust::host_vector<int> h_result(12);
  typedef thrust::host_vector<int>::iterator Iterator;

  Iterator result_end = thrust::set_union(
      h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());

  int ref[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12};
  for (int i = 0; i < 8; i++) {
    if (h_result[i] != ref[i]) {
      printf("test_10 run failed\n");
      exit(-1);
    }
  }

  printf("test_10 run pass\n");
}

void test_11() {
  int A1[7] = {12, 10, 8, 6, 4, 2, 0};
  int A2[5] = {9, 7, 5, 3, 1};

  thrust::host_vector<int> h_A1(A1, A1 + 7);
  thrust::host_vector<int> h_A2(A2, A2 + 5);
  thrust::host_vector<int> h_result(12);
  typedef thrust::host_vector<int>::iterator Iterator;

  Iterator result_end =
      thrust::set_union(thrust::host, h_A1.begin(), h_A1.end(), h_A2.begin(),
                        h_A2.end(), h_result.begin(), thrust::greater<int>());

  int ref[12] = {12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  for (int i = 0; i < 8; i++) {
    if (h_result[i] != ref[i]) {
      printf("test_11 run failed\n");
      exit(-1);
    }
  }

  printf("test_11 run pass\n");
}

void test_12() {
  int A1[7] = {12, 10, 8, 6, 4, 2, 0};
  int A2[5] = {9, 7, 5, 3, 1};

  thrust::host_vector<int> h_A1(A1, A1 + 7);
  thrust::host_vector<int> h_A2(A2, A2 + 5);
  thrust::host_vector<int> h_result(12);
  typedef thrust::host_vector<int>::iterator Iterator;

  Iterator result_end =
      thrust::set_union(h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(),
                        h_result.begin(), thrust::greater<int>());

  int ref[12] = {12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  for (int i = 0; i < 8; i++) {
    if (h_result[i] != ref[i]) {
      printf("test_12 run failed\n");
      exit(-1);
    }
  }

  printf("test_12 run pass\n");
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
