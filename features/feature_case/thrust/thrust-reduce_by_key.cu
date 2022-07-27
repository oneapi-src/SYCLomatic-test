// ====------ thrust-reduce_by_key.cu -------------------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

void test_1() {
  int keys_in[7] =   {3,  3,  2,  3,  1,  1,  2};
  int values_in[7] = {1,  2,  3,  4,  5,  6,  7};
  thrust::device_vector<int> d_keys_in(keys_in, keys_in + 7);
  thrust::device_vector<int> d_values_in(values_in, values_in + 7);
  thrust::device_vector<int> d_keys_out(5);
  thrust::device_vector<int> d_values_out(5);
  thrust::reduce_by_key(thrust::device, d_keys_in.begin(), d_keys_in.end(), d_values_in.begin(), d_keys_out.begin(), d_values_out.begin());
  
  int keys_ref[5] = {3, 2, 3, 1, 2};
  int values_ref[5] = {3, 3, 4, 11, 7};

  for (int i = 0; i < 5; i++) {
    if ((d_keys_out[i] != keys_ref[i]) || (values_ref[i] != d_values_out[i])) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }
  printf("test_1 run passed!\n");
}

void test_2() {
  int keys_in[7] =   {3,  3,  2,  3,  1,  1,  2};
  int values_in[7] = {1,  2,  3,  4,  5,  6,  7};
  thrust::device_vector<int> d_keys_in(keys_in, keys_in + 7);
  thrust::device_vector<int> d_values_in(values_in, values_in + 7);
  thrust::device_vector<int> d_keys_out(5);
  thrust::device_vector<int> d_values_out(5);
  thrust::reduce_by_key(d_keys_in.begin(), d_keys_in.end(), d_values_in.begin(), d_keys_out.begin(), d_values_out.begin());
  
  int keys_ref[5] = {3, 2, 3, 1, 2};
  int values_ref[5] = {3, 3, 4, 11, 7};

  for (int i = 0; i < 5; i++) {
    if ((d_keys_out[i] != keys_ref[i]) || (values_ref[i] != d_values_out[i])) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }
  printf("test_2 run passed!\n");
}

void test_3() {
  int keys_in[7] =   {3,  3,  2,  3,  1,  1,  2};
  int values_in[7] = {1,  2,  3,  4,  5,  6,  7};
  thrust::device_vector<int> d_keys_in(keys_in, keys_in + 7);
  thrust::device_vector<int> d_values_in(values_in, values_in + 7);
  thrust::device_vector<int> d_keys_out(5);
  thrust::device_vector<int> d_values_out(5);
  thrust::reduce_by_key(thrust::device, d_keys_in.begin(), d_keys_in.end(), d_values_in.begin(), d_keys_out.begin(), d_values_out.begin(), thrust::equal_to<int>());
  
  int keys_ref[5] = {3, 2, 3, 1, 2};
  int values_ref[5] = {3, 3, 4, 11, 7};

  for (int i = 0; i < 5; i++) {
    if ((d_keys_out[i] != keys_ref[i]) || (values_ref[i] != d_values_out[i])) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }
  printf("test_3 run passed!\n");
}

void test_4() {
  int keys_in[7] =   {3,  3,  2,  3,  1,  1,  2};
  thrust::device_vector<int> d_keys_in(keys_in, keys_in + 7);
  thrust::device_vector<int> d_keys_out(5);
  thrust::device_vector<int> d_values_out(5);
  thrust::reduce_by_key(d_keys_in.begin(), d_keys_in.end(), thrust::constant_iterator<int>(1), d_keys_out.begin(), d_values_out.begin(), thrust::equal_to<int>());
  
  int keys_ref[5] = {3, 2, 3, 1, 2};
  int values_ref[5] = {2, 1, 1, 2, 1};

  for (int i = 0; i < 5; i++) {
    if ((d_keys_out[i] != keys_ref[i]) || (values_ref[i] != d_values_out[i])) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }
  printf("test_4 run passed!\n");
}

void test_5() {
  int keys_in[7] =   {3,  3,  2,  3,  1,  1,  2};
  thrust::device_vector<int> d_keys_in(keys_in, keys_in + 7);
  thrust::device_vector<int> d_keys_out(5);
  thrust::device_vector<int> d_values_out(5);
  thrust::reduce_by_key(thrust::device, d_keys_in.begin(), d_keys_in.end(), thrust::constant_iterator<int>(1), d_keys_out.begin(), d_values_out.begin(), thrust::equal_to<int>(), thrust::plus<int>());
  
  int keys_ref[5] = {3, 2, 3, 1, 2};
  int values_ref[5] = {2, 1, 1, 2, 1};

  for (int i = 0; i < 5; i++) {
    if ((d_keys_out[i] != keys_ref[i]) || (values_ref[i] != d_values_out[i])) {
      printf("test_5 run failed\n");
      exit(-1);
    }
  }
  printf("test_5 run passed!\n");
}

void test_6() {
  int keys_in[7] =   {3,  3,  2,  3,  1,  1,  2};
  int values_in[7] = {1,  2,  3,  4,  5,  6,  7};
  thrust::device_vector<int> d_keys_in(keys_in, keys_in + 7);
  thrust::device_vector<int> d_values_in(values_in, values_in + 7);
  thrust::device_vector<int> d_keys_out(5);
  thrust::device_vector<int> d_values_out(5);
  thrust::reduce_by_key(d_keys_in.begin(), d_keys_in.end(), d_values_in.begin(), d_keys_out.begin(), d_values_out.begin(), thrust::equal_to<int>(), thrust::plus<int>());
  
  int keys_ref[5] = {3, 2, 3, 1, 2};
  int values_ref[5] = {3, 3, 4, 11, 7};

  for (int i = 0; i < 5; i++) {
    if ((d_keys_out[i] != keys_ref[i]) || (values_ref[i] != d_values_out[i])) {
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
