// ====------ thrust-unique_by_key_copy.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

void test_1() {
  const int N = 7;
  int A[N] = {1, 2, 2, 3, 3, 3, 4};
  int B[N] = {1, 2, 3, 4, 5, 6, 7};

  thrust::device_vector<int> d_keys(A, A + N);
  thrust::device_vector<int> d_values(B, B + N);
  thrust::device_vector<int> d_output_keys(N);
  thrust::device_vector<int> d_output_values(N);
  thrust::equal_to<int> binary_pred;

  typedef thrust::pair<thrust::device_vector<int>::iterator,
                       thrust::device_vector<int>::iterator>
      iter_pair;

  iter_pair new_last = thrust::unique_by_key_copy(
      thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(),
      d_output_keys.begin(), d_output_values.begin(), binary_pred);

  int keys_ref[10] = {1, 2, 3, 4};
  int values_ref[10] = {1, 2, 4, 7};
  for (int i = 0; i < 4; i++) {
    if (d_output_keys[i] != keys_ref[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }

  for (int i = 0; i < 4; i++) {
    if (d_output_values[i] != values_ref[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }

  if (new_last.first - d_output_keys.begin() != 4 ||
      new_last.second - d_output_values.begin() != 4) {
    printf("test_1 run failed\n");
    exit(-1);
  }

  printf("test_1 run passed!\n");
}

void test_2() {

  const int N = 7;
  int A[N] = {1, 2, 2, 3, 3, 3, 4};
  int B[N] = {1, 2, 3, 4, 5, 6, 7};

  thrust::host_vector<int> h_keys(A, A + N);
  thrust::host_vector<int> h_values(B, B + N);
  thrust::host_vector<int> h_output_keys(N);
  thrust::host_vector<int> h_output_values(N);
  thrust::equal_to<int> binary_pred;

  typedef thrust::pair<thrust::host_vector<int>::iterator,
                       thrust::host_vector<int>::iterator>
      iter_pair;

  iter_pair new_last = thrust::unique_by_key_copy(
      thrust::seq, h_keys.begin(), h_keys.end(), h_values.begin(),
      h_output_keys.begin(), h_output_values.begin(), binary_pred);

  int keys_ref[10] = {1, 2, 3, 4};
  int values_ref[10] = {1, 2, 4, 7};
  for (int i = 0; i < 4; i++) {
    if (h_output_keys[i] != keys_ref[i]) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }

  for (int i = 0; i < 4; i++) {
    if (h_output_values[i] != values_ref[i]) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }

  if (new_last.first - h_output_keys.begin() != 4 ||
      new_last.second - h_output_values.begin() != 4) {
    printf("test_2 run failed\n");
    exit(-1);
  }

  printf("test_2 run passed!\n");
}

void test_3() {

  const int N = 7;
  int A[N] = {1, 2, 2, 3, 3, 3, 4};
  int B[N] = {1, 2, 3, 4, 5, 6, 7};

  thrust::device_vector<int> d_keys(A, A + N);
  thrust::device_vector<int> d_values(B, B + N);
  thrust::device_vector<int> d_output_keys(N);
  thrust::device_vector<int> d_output_values(N);
  thrust::equal_to<int> binary_pred;

  typedef thrust::pair<thrust::device_vector<int>::iterator,
                       thrust::device_vector<int>::iterator>
      iter_pair;

  iter_pair new_last = thrust::unique_by_key_copy(
      d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(),
      d_output_values.begin(), binary_pred);

  int keys_ref[10] = {1, 2, 3, 4};
  int values_ref[10] = {1, 2, 4, 7};
  for (int i = 0; i < 4; i++) {
    if (d_output_keys[i] != keys_ref[i]) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }

  for (int i = 0; i < 4; i++) {
    if (d_output_values[i] != values_ref[i]) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }

  if (new_last.first - d_output_keys.begin() != 4 ||
      new_last.second - d_output_values.begin() != 4) {
    printf("test_3 run failed\n");
    exit(-1);
  }

  printf("test_3 run passed!\n");
}

void test_4() {

  const int N = 7;
  int A[N] = {1, 2, 2, 3, 3, 3, 4};
  int B[N] = {1, 2, 3, 4, 5, 6, 7};

  thrust::host_vector<int> h_keys(A, A + N);
  thrust::host_vector<int> h_values(B, B + N);
  thrust::host_vector<int> h_output_keys(N);
  thrust::host_vector<int> h_output_values(N);
  thrust::equal_to<int> binary_pred;

  typedef thrust::pair<thrust::host_vector<int>::iterator,
                       thrust::host_vector<int>::iterator>
      iter_pair;

  iter_pair new_last = thrust::unique_by_key_copy(
      h_keys.begin(), h_keys.end(), h_values.begin(), h_output_keys.begin(),
      h_output_values.begin(), binary_pred);

  int keys_ref[10] = {1, 2, 3, 4};
  int values_ref[10] = {1, 2, 4, 7};
  for (int i = 0; i < 4; i++) {
    if (h_output_keys[i] != keys_ref[i]) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }

  for (int i = 0; i < 4; i++) {
    if (h_output_values[i] != values_ref[i]) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }

  if (new_last.first - h_output_keys.begin() != 4 ||
      new_last.second - h_output_values.begin() != 4) {
    printf("test_3 run failed\n");
    exit(-1);
  }

  printf("test_4 run passed!\n");
}

void test_5() {

  const int N = 7;
  int A[N] = {1, 2, 2, 3, 3, 3, 4};
  int B[N] = {1, 2, 3, 4, 5, 6, 7};

  thrust::host_vector<int> h_keys(A, A + N);
  thrust::host_vector<int> h_values(B, B + N);
  thrust::host_vector<int> h_output_keys(N);
  thrust::host_vector<int> h_output_values(N);

  typedef thrust::pair<thrust::host_vector<int>::iterator,
                       thrust::host_vector<int>::iterator>
      iter_pair;

  iter_pair new_last = thrust::unique_by_key_copy(
      h_keys.begin(), h_keys.end(), h_values.begin(), h_output_keys.begin(),
      h_output_values.begin());

  int keys_ref[10] = {1, 2, 3, 4};
  int values_ref[10] = {1, 2, 4, 7};
  for (int i = 0; i < 4; i++) {
    if (h_output_keys[i] != keys_ref[i]) {
      printf("test_5 run failed\n");
      exit(-1);
    }
  }

  for (int i = 0; i < 4; i++) {
    if (h_output_values[i] != values_ref[i]) {
      printf("test_5 run failed\n");
      exit(-1);
    }
  }

  if (new_last.first - h_output_keys.begin() != 4 ||
      new_last.second - h_output_values.begin() != 4) {
    printf("test_5 run failed\n");
    exit(-1);
  }

  printf("test_5 run passed!\n");
}

void test_6() {

  const int N = 7;
  int A[N] = {1, 2, 2, 3, 3, 3, 4};
  int B[N] = {1, 2, 3, 4, 5, 6, 7};

  thrust::device_vector<int> d_keys(A, A + N);
  thrust::device_vector<int> d_values(B, B + N);
  thrust::device_vector<int> d_output_keys(N);
  thrust::device_vector<int> d_output_values(N);

  typedef thrust::pair<thrust::device_vector<int>::iterator,
                       thrust::device_vector<int>::iterator>
      iter_pair;

  iter_pair new_last = thrust::unique_by_key_copy(
      d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(),
      d_output_values.begin());

  int keys_ref[10] = {1, 2, 3, 4};
  int values_ref[10] = {1, 2, 4, 7};
  for (int i = 0; i < 4; i++) {
    if (d_output_keys[i] != keys_ref[i]) {
      printf("test_6 run failed\n");
      exit(-1);
    }
  }

  for (int i = 0; i < 4; i++) {
    if (d_output_values[i] != values_ref[i]) {
      printf("test_6 run failed\n");
      exit(-1);
    }
  }

  if (new_last.first - d_output_keys.begin() != 4 ||
      new_last.second - d_output_values.begin() != 4) {
    printf("test_6 run failed\n");
    exit(-1);
  }

  printf("test_6 run passed!\n");
}

void test_7() {

  const int N = 7;
  int A[N] = {1, 2, 2, 3, 3, 3, 4};
  int B[N] = {1, 2, 3, 4, 5, 6, 7};

  thrust::device_vector<int> d_keys(A, A + N);
  thrust::device_vector<int> d_values(B, B + N);
  thrust::device_vector<int> d_output_keys(N);
  thrust::device_vector<int> d_output_values(N);

  typedef thrust::pair<thrust::device_vector<int>::iterator,
                       thrust::device_vector<int>::iterator>
      iter_pair;

  thrust::pair<int *, int *> new_end;

  iter_pair new_last = thrust::unique_by_key_copy(
      thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(),
      d_output_keys.begin(), d_output_values.begin());

  int keys_ref[10] = {1, 2, 3, 4};
  int values_ref[10] = {1, 2, 4, 7};
  for (int i = 0; i < 4; i++) {
    if (d_output_keys[i] != keys_ref[i]) {
      printf("test_7 run failed\n");
      exit(-1);
    }
  }

  for (int i = 0; i < 4; i++) {
    if (d_output_values[i] != values_ref[i]) {
      printf("test_7 run failed\n");
      exit(-1);
    }
  }

  if (new_last.first - d_output_keys.begin() != 4 ||
      new_last.second - d_output_values.begin() != 4) {
    printf("test_7 run failed\n");
    exit(-1);
  }

  printf("test_7 run passed!\n");
}

void test_8() {

  const int N = 7;
  int A[N] = {1, 2, 2, 3, 3, 3, 4};
  int B[N] = {1, 2, 3, 4, 5, 6, 7};

  thrust::host_vector<int> h_keys(A, A + N);
  thrust::host_vector<int> h_values(B, B + N);
  thrust::host_vector<int> h_output_keys(N);
  thrust::host_vector<int> h_output_values(N);

  typedef thrust::pair<thrust::host_vector<int>::iterator,
                       thrust::host_vector<int>::iterator>
      iter_pair;

  iter_pair new_last = thrust::unique_by_key_copy(
      thrust::seq, h_keys.begin(), h_keys.end(), h_values.begin(),
      h_output_keys.begin(), h_output_values.begin());

  int keys_ref[10] = {1, 2, 3, 4};
  int values_ref[10] = {1, 2, 4, 7};
  for (int i = 0; i < 4; i++) {
    if (h_output_keys[i] != keys_ref[i]) {
      printf("test_8 run failed\n");
      exit(-1);
    }
  }

  for (int i = 0; i < 4; i++) {
    if (h_output_values[i] != values_ref[i]) {
      printf("test_8 run failed\n");
      exit(-1);
    }
  }

  if (new_last.first - h_output_keys.begin() != 4 ||
      new_last.second - h_output_values.begin() != 4) {
    printf("test_8 run failed\n");
    exit(-1);
  }

  printf("test_8 run passed!\n");
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

  return 0;
}
