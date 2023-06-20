// ====------ thrust_set_intersection.cu--------------- *- CUDA -*---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/set_operations.h>

void test_1() {
  int A1[6] = {1, 3, 5, 7, 9, 11};
  int A2[7] = {1, 1, 2, 3, 5, 8, 13};
  int result[3];
  int *result_end =
      thrust::set_intersection(thrust::host, A1, A1 + 6, A2, A2 + 7, result);

  int ref[3] = {1, 3, 5};

  for (int i = 0; i < 3; i++) {
    if (result[i] != ref[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }

  if (result_end - result != 3) {
    printf("test_1 run failed\n");
    exit(-1);
  }

  printf("test_1 passed!\n");
}

void test_2() {
  int A1[6] = {1, 3, 5, 7, 9, 11};
  int A2[7] = {1, 1, 2, 3, 5, 8, 13};
  int result[3];
  int *result_end = thrust::set_intersection(A1, A1 + 6, A2, A2 + 7, result);

  int ref[3] = {1, 3, 5};

  for (int i = 0; i < 3; i++) {
    if (result[i] != ref[i]) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }

  if (result_end - result != 3) {
    printf("test_2 run failed\n");
    exit(-1);
  }

  printf("test_2 passed!\n");
}

void test_3() {

  int A1[6] = {11, 9, 7, 5, 3, 1};
  int A2[7] = {13, 8, 5, 3, 2, 1, 1};
  int result[3];
  int *result_end = thrust::set_intersection(
      thrust::host, A1, A1 + 6, A2, A2 + 7, result, thrust::greater<int>());

  int ref[3] = {5, 3, 1};

  for (int i = 0; i < 3; i++) {
    if (result[i] != ref[i]) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }

  if (result_end - result != 3) {
    printf("test_3 run failed\n");
    exit(-1);
  }

  printf("test_3 passed!\n");
}

void test_4() {

  int A1[6] = {11, 9, 7, 5, 3, 1};
  int A2[7] = {13, 8, 5, 3, 2, 1, 1};
  int result[3];
  int *result_end = thrust::set_intersection(
      thrust::host, A1, A1 + 6, A2, A2 + 7, result, thrust::greater<int>());

  int ref[3] = {5, 3, 1};

  for (int i = 0; i < 3; i++) {
    if (result[i] != ref[i]) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }

  if (result_end - result != 3) {
    printf("test_4 run failed\n");
    exit(-1);
  }

  printf("test_4 passed!\n");
}

void test_5() {
  int A1[6] = {1, 3, 5, 7, 9, 11};
  int A2[7] = {1, 1, 2, 3, 5, 8, 13};

  thrust::device_vector<int> d_A1(A1, A1 + 6);
  thrust::device_vector<int> d_A2(A2, A2 + 7);
  thrust::device_vector<int> d_result(3);
  thrust::device_vector<int>::iterator end;

  end = thrust::set_intersection(thrust::device, d_A1.begin(), d_A1.end(),
                                 d_A2.begin(), d_A2.end(), d_result.begin());

  int ref[3] = {1, 3, 5};

  for (int i = 0; i < 3; i++) {
    if (d_result[i] != ref[i]) {
      printf("test_5 run failed\n");
      exit(-1);
    }
  }

  if (end - d_result.begin() != 3) {
    printf("test_5 run failed\n");
    exit(-1);
  }

  printf("test_5 passed!\n");
}

void test_6() {
  int A1[6] = {1, 3, 5, 7, 9, 11};
  int A2[7] = {1, 1, 2, 3, 5, 8, 13};

  thrust::device_vector<int> d_A1(A1, A1 + 6);
  thrust::device_vector<int> d_A2(A2, A2 + 7);
  thrust::device_vector<int> d_result(3);
  thrust::device_vector<int>::iterator end;

  end = thrust::set_intersection(d_A1.begin(), d_A1.end(), d_A2.begin(),
                                 d_A2.end(), d_result.begin());

  int ref[3] = {1, 3, 5};

  for (int i = 0; i < 3; i++) {
    if (d_result[i] != ref[i]) {
      printf("test_6 run failed\n");
      exit(-1);
    }
  }

  if (end - d_result.begin() != 3) {
    printf("test_6 run failed\n");
    exit(-1);
  }

  printf("test_6 passed!\n");
}

void test_7() {

  int A1[6] = {11, 9, 7, 5, 3, 1};
  int A2[7] = {13, 8, 5, 3, 2, 1, 1};

  thrust::device_vector<int> d_A1(A1, A1 + 6);
  thrust::device_vector<int> d_A2(A2, A2 + 7);
  thrust::device_vector<int> d_result(3);
  thrust::device_vector<int>::iterator end;

  end = thrust::set_intersection(thrust::device, d_A1.begin(), d_A1.end(),
                                 d_A2.begin(), d_A2.end(), d_result.begin(),
                                 thrust::greater<int>());

  int ref[3] = {5, 3, 1};

  for (int i = 0; i < 3; i++) {
    if (d_result[i] != ref[i]) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }

  if (end - d_result.begin() != 3) {
    printf("test_7 run failed\n");
    exit(-1);
  }

  printf("test_7 passed!\n");
}

void test_8() {

  int A1[6] = {11, 9, 7, 5, 3, 1};
  int A2[7] = {13, 8, 5, 3, 2, 1, 1};

  thrust::device_vector<int> d_A1(A1, A1 + 6);
  thrust::device_vector<int> d_A2(A2, A2 + 7);
  thrust::device_vector<int> d_result(3);
  thrust::device_vector<int>::iterator end;

  end = thrust::set_intersection(d_A1.begin(), d_A1.end(), d_A2.begin(),
                                 d_A2.end(), d_result.begin(),
                                 thrust::greater<int>());

  int ref[3] = {5, 3, 1};

  for (int i = 0; i < 3; i++) {
    if (d_result[i] != ref[i]) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }

  if (end - d_result.begin() != 3) {
    printf("test_8 run failed\n");
    exit(-1);
  }

  printf("test_8 passed!\n");
}

void test_9() {
  int A1[6] = {1, 3, 5, 7, 9, 11};
  int A2[7] = {1, 1, 2, 3, 5, 8, 13};

  thrust::host_vector<int> h_A1(A1, A1 + 6);
  thrust::host_vector<int> h_A2(A2, A2 + 7);
  thrust::host_vector<int> h_result(3);
  thrust::host_vector<int>::iterator end;

  end = thrust::set_intersection(thrust::host, h_A1.begin(), h_A1.end(),
                                 h_A2.begin(), h_A2.end(), h_result.begin());

  int ref[3] = {1, 3, 5};

  for (int i = 0; i < 3; i++) {
    if (h_result[i] != ref[i]) {
      printf("test_9 run failed\n");
      exit(-1);
    }
  }

  if (end - h_result.begin() != 3) {
    printf("test_9 run failed\n");
    exit(-1);
  }

  printf("test_9 passed!\n");
}

void test_10() {
  int A1[6] = {1, 3, 5, 7, 9, 11};
  int A2[7] = {1, 1, 2, 3, 5, 8, 13};

  thrust::host_vector<int> h_A1(A1, A1 + 6);
  thrust::host_vector<int> h_A2(A2, A2 + 7);
  thrust::host_vector<int> h_result(3);
  thrust::host_vector<int>::iterator end;

  end = thrust::set_intersection(h_A1.begin(), h_A1.end(), h_A2.begin(),
                                 h_A2.end(), h_result.begin());

  int ref[3] = {1, 3, 5};

  for (int i = 0; i < 3; i++) {
    if (h_result[i] != ref[i]) {
      printf("test_10 run failed\n");
      exit(-1);
    }
  }

  if (end - h_result.begin() != 3) {
    printf("test_10 run failed\n");
    exit(-1);
  }

  printf("test_10 passed!\n");
}

void test_11() {

  int A1[6] = {11, 9, 7, 5, 3, 1};
  int A2[7] = {13, 8, 5, 3, 2, 1, 1};

  thrust::host_vector<int> h_A1(A1, A1 + 6);
  thrust::host_vector<int> h_A2(A2, A2 + 7);
  thrust::host_vector<int> h_result(3);
  thrust::host_vector<int>::iterator end;

  end = thrust::set_intersection(thrust::host, h_A1.begin(), h_A1.end(),
                                 h_A2.begin(), h_A2.end(), h_result.begin(),
                                 thrust::greater<int>());

  int ref[3] = {5, 3, 1};

  for (int i = 0; i < 3; i++) {
    if (h_result[i] != ref[i]) {
      printf("test_11 run failed\n");
      exit(-1);
    }
  }

  if (end - h_result.begin() != 3) {
    printf("test_11 run failed\n");
    exit(-1);
  }

  printf("test_11 passed!\n");
}

void test_12() {

  int A1[6] = {11, 9, 7, 5, 3, 1};
  int A2[7] = {13, 8, 5, 3, 2, 1, 1};

  thrust::host_vector<int> h_A1(A1, A1 + 6);
  thrust::host_vector<int> h_A2(A2, A2 + 7);
  thrust::host_vector<int> h_result(3);
  thrust::host_vector<int>::iterator end;

  end = thrust::set_intersection(h_A1.begin(), h_A1.end(), h_A2.begin(),
                                 h_A2.end(), h_result.begin(),
                                 thrust::greater<int>());

  int ref[3] = {5, 3, 1};

  for (int i = 0; i < 3; i++) {
    if (h_result[i] != ref[i]) {
      printf("test_12 run failed\n");
      exit(-1);
    }
  }

  if (end - h_result.begin() != 3) {
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
