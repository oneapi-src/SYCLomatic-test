// ====------ thrust_swap_ranges.cu------------- *- CUDA -* --------------===//
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
#include <thrust/uninitialized_fill.h>

void test_1() {

  thrust::device_vector<int> d_v1(2), d_v2(2);
  d_v1[0] = 1;
  d_v1[1] = 2;
  d_v2[0] = 3;
  d_v2[1] = 4;
  thrust::swap_ranges(thrust::device, d_v1.begin(), d_v1.end(), d_v2.begin());

  int ref1[2] = {3, 4};
  int ref2[2] = {1, 2};

  for (int i = 0; i < 2; i++) {
    if (d_v1[i] != ref1[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }

  for (int i = 0; i < 2; i++) {
    if (d_v2[i] != ref2[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }

  printf("test_1 run passed!\n");
}

void test_2() {

  thrust::device_vector<int> d_v1(2), d_v2(2);
  d_v1[0] = 1;
  d_v1[1] = 2;
  d_v2[0] = 3;
  d_v2[1] = 4;
  thrust::swap_ranges(d_v1.begin(), d_v1.end(), d_v2.begin());

  int ref1[2] = {3, 4};
  int ref2[2] = {1, 2};

  for (int i = 0; i < 2; i++) {
    if (d_v1[i] != ref1[i]) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }

  for (int i = 0; i < 2; i++) {
    if (d_v2[i] != ref2[i]) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }

  printf("test_2 run passed!\n");
}



void test_3() {

  thrust::host_vector<int> h_v1(2), h_v2(2);
  h_v1[0] = 1;
  h_v1[1] = 2;
  h_v2[0] = 3;
  h_v2[1] = 4;
  thrust::swap_ranges(thrust::host, h_v1.begin(), h_v1.end(), h_v2.begin());

  int ref1[2] = {3, 4};
  int ref2[2] = {1, 2};

  for (int i = 0; i < 2; i++) {
    if (h_v1[i] != ref1[i]) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }

  for (int i = 0; i < 2; i++) {
    if (h_v2[i] != ref2[i]) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }

  printf("test_3 run passed!\n");
}





void test_4() {

  thrust::host_vector<int> h_v1(2), h_v2(2);
  h_v1[0] = 1;
  h_v1[1] = 2;
  h_v2[0] = 3;
  h_v2[1] = 4;
  thrust::swap_ranges(h_v1.begin(), h_v1.end(), h_v2.begin());

  int ref1[2] = {3, 4};
  int ref2[2] = {1, 2};

  for (int i = 0; i < 2; i++) {
    if (h_v1[i] != ref1[i]) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }

  for (int i = 0; i < 2; i++) {
    if (h_v2[i] != ref2[i]) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }

  printf("test_4 run passed!\n");
}

void test_5() {

  int v1[2], v2[2];
  v1[0] = 1;
  v1[1] = 2;
  v2[0] = 3;
  v2[1] = 4;
  thrust::swap_ranges(thrust::host, v1, v1 + 2, v2);

  int ref1[2] = {3, 4};
  int ref2[2] = {1, 2};

  for (int i = 0; i < 2; i++) {
    if (v1[i] != ref1[i]) {
      printf("test_5 run failed\n");
      exit(-1);
    }
  }

  for (int i = 0; i < 2; i++) {
    if (v2[i] != ref2[i]) {
      printf("test_5 run failed\n");
      exit(-1);
    }
  }

  printf("test_5 run passed!\n");
}

void test_6() {

  int v1[2], v2[2];
  v1[0] = 1;
  v1[1] = 2;
  v2[0] = 3;
  v2[1] = 4;
  thrust::swap_ranges(v1, v1 + 2, v2);

  int ref1[2] = {3, 4};
  int ref2[2] = {1, 2};

  for (int i = 0; i < 2; i++) {
    if (v1[i] != ref1[i]) {
      printf("test_6 run failed\n");
      exit(-1);
    }
  }

  for (int i = 0; i < 2; i++) {
    if (v2[i] != ref2[i]) {
      printf("test_6 run failed\n");
      exit(-1);
    }
  }

  printf("test_6 run passed!\n");
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