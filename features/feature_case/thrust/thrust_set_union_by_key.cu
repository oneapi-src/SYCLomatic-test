// ====------ thrust_set_union_by_key.cu--------------- *- CUDA -* --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/set_operations.h>

void test_1() {
  int A_keys[7] = {0, 2, 4};
  int A_vals[7] = {0, 0, 0};
  int B_keys[5] = {0, 3, 3, 4};
  int B_vals[5] = {1, 1, 1, 1};
  int keys_result[5];
  int vals_result[5];
  thrust::pair<int *, int *> end = thrust::set_union_by_key(
      thrust::host, A_keys, A_keys + 3, B_keys, B_keys + 4, A_vals, B_vals,
      keys_result, vals_result);

  int ref_keys[5] = {0, 2, 3, 3, 4};
  int ref_vals[5] = {0, 0, 1, 1, 0};
  for (int i = 0; i < 5; i++) {
    if (keys_result[i] != ref_keys[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < 5; i++) {
    if (vals_result[i] != ref_vals[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }
  printf("test_1 run pass!\n");
}

void test_2() {
  int A_keys[7] = {0, 2, 4};
  int A_vals[7] = {0, 0, 0};
  int B_keys[5] = {0, 3, 3, 4};
  int B_vals[5] = {1, 1, 1, 1};
  int keys_result[5];
  int vals_result[5];
  thrust::pair<int *, int *> end =
      thrust::set_union_by_key(A_keys, A_keys + 3, B_keys, B_keys + 4, A_vals,
                               B_vals, keys_result, vals_result);

  int ref_keys[5] = {0, 2, 3, 3, 4};
  int ref_vals[5] = {0, 0, 1, 1, 0};
  for (int i = 0; i < 5; i++) {
    if (keys_result[i] != ref_keys[i]) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < 5; i++) {
    if (vals_result[i] != ref_vals[i]) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }
  printf("test_2 run pass!\n");
}

struct Compare {
  __host__ __device__ bool operator()(const int &a, const int &b) {
    // Custom comparison function
    // Returns true if a < b, false otherwise
    return a < b;
  }
};

void test_3() {

  int A_keys[7] = {0, 1, 2, 2, 4, 6, 7};
  int A_vals[7] = {0, 0, 0, 0, 0, 0, 0};
  int B_keys[5] = {1, 1, 2, 5, 8};
  int B_vals[5] = {1, 1, 1, 1, 1};
  int keys_result[20];
  int vals_result[20];

  thrust::pair<int *, int *> end = thrust::set_union_by_key(
      thrust::host, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals,
      keys_result, vals_result, Compare());

  int ref_keys[10] = {0, 1, 1, 2, 2, 4, 5, 6, 7, 8};
  int ref_vals[10] = {0, 0, 1, 0, 0, 0, 1, 0, 0, 1};
  for (int i = 0; i < 10; i++) {
    if (keys_result[i] != ref_keys[i]) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < 10; i++) {
    if (vals_result[i] != ref_vals[i]) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }

  printf("test_3 run pass!\n");
}

void test_4() {
  int A_keys[7] = {0, 1, 2, 2, 4, 6, 7};
  int A_vals[7] = {0, 0, 0, 0, 0, 0, 0};
  int B_keys[5] = {1, 1, 2, 5, 8};
  int B_vals[5] = {1, 1, 1, 1, 1};
  int keys_result[20];
  int vals_result[20];

  thrust::pair<int *, int *> end =
      thrust::set_union_by_key(A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals,
                               B_vals, keys_result, vals_result, Compare());

  int ref_keys[10] = {0, 1, 1, 2, 2, 4, 5, 6, 7, 8};
  int ref_vals[10] = {0, 0, 1, 0, 0, 0, 1, 0, 0, 1};
  for (int i = 0; i < 10; i++) {
    if (keys_result[i] != ref_keys[i]) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < 10; i++) {
    if (vals_result[i] != ref_vals[i]) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }

  printf("test_4 run pass\n");
}

void test_5() {
  int A_keys[7] = {0, 1, 2, 2, 4, 6, 7};
  int A_vals[7] = {0, 0, 0, 0, 0, 0, 0};
  int B_keys[5] = {1, 1, 2, 5, 8};
  int B_vals[5] = {1, 1, 1, 1, 1};
  thrust::device_vector<int> d_keys_result(10);
  thrust::device_vector<int> d_vals_result(10);

  thrust::device_vector<int> d_A_keys(A_keys, A_keys + 7);
  thrust::device_vector<int> d_A_vals(A_vals, A_vals + 7);
  thrust::device_vector<int> d_B_keys(B_keys, B_keys + 5);
  thrust::device_vector<int> d_B_vals(B_vals, B_vals + 5);

  typedef thrust::device_vector<int>::iterator Iterator;

  thrust::pair<Iterator, Iterator> d_result;

  d_result = thrust::set_union_by_key(
      thrust::device, d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(),
      d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(),
      d_vals_result.begin());

  int ref_keys[10] = {0, 1, 1, 2, 2, 4, 5, 6, 7, 8};
  int ref_vals[10] = {
      0, 0, 1, 0, 0, 0, 1, 0, 0, 1,
  };
  for (int i = 0; i < 10; i++) {
    if (d_keys_result[i] != ref_keys[i]) {
      printf("test_5 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < 10; i++) {
    if (d_vals_result[i] != ref_vals[i]) {
      printf("test_5 run failed\n");
      exit(-1);
    }
  }
  printf("test_5 run pass!\n");
}

void test_6() {
  int A_keys[7] = {0, 1, 2, 2, 4, 6, 7};
  int A_vals[7] = {0, 0, 0, 0, 0, 0, 0};
  int B_keys[5] = {1, 1, 2, 5, 8};
  int B_vals[5] = {1, 1, 1, 1, 1};
  thrust::device_vector<int> d_keys_result(10);
  thrust::device_vector<int> d_vals_result(10);

  thrust::device_vector<int> d_A_keys(A_keys, A_keys + 7);
  thrust::device_vector<int> d_A_vals(A_vals, A_vals + 7);
  thrust::device_vector<int> d_B_keys(B_keys, B_keys + 5);
  thrust::device_vector<int> d_B_vals(B_vals, B_vals + 5);

  typedef thrust::device_vector<int>::iterator Iterator;

  thrust::pair<Iterator, Iterator> d_result;

  d_result = thrust::set_union_by_key(
      d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(),
      d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(),
      d_vals_result.begin());

  int ref_keys[10] = {0, 1, 1, 2, 2, 4, 5, 6, 7, 8};
  int ref_vals[10] = {
      0, 0, 1, 0, 0, 0, 1, 0, 0, 1,
  };
  for (int i = 0; i < 10; i++) {
    if (d_keys_result[i] != ref_keys[i]) {
      printf("test_6 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < 10; i++) {
    if (d_vals_result[i] != ref_vals[i]) {
      printf("test_6 run failed\n");
      exit(-1);
    }
  }
  printf("test_6 run pass\n");
}

void test_7() {
  int A_keys[7] = {0, 1, 2, 2, 4, 6, 7};
  int A_vals[7] = {0, 0, 0, 0, 0, 0, 0};
  int B_keys[5] = {1, 1, 2, 5, 8};
  int B_vals[5] = {1, 1, 1, 1, 1};
  thrust::device_vector<int> d_keys_result(10);
  thrust::device_vector<int> d_vals_result(10);

  thrust::device_vector<int> d_A_keys(A_keys, A_keys + 7);
  thrust::device_vector<int> d_A_vals(A_vals, A_vals + 7);
  thrust::device_vector<int> d_B_keys(B_keys, B_keys + 5);
  thrust::device_vector<int> d_B_vals(B_vals, B_vals + 5);

  typedef thrust::device_vector<int>::iterator Iterator;

  thrust::pair<Iterator, Iterator> d_result;

  d_result = thrust::set_union_by_key(
      thrust::device, d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(),
      d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(),
      d_vals_result.begin(), Compare());

  int ref_keys[10] = {0, 1, 1, 2, 2, 4, 5, 6, 7, 8};
  int ref_vals[10] = {0, 0, 1, 0, 0, 0, 1, 0, 0, 1};
  for (int i = 0; i < 10; i++) {
    if (d_keys_result[i] != ref_keys[i]) {
      printf("test_7 run failed\n");
      exit(-1);
    }
  }

  for (int i = 0; i < 10; i++) {
    if (d_vals_result[i] != ref_vals[i]) {
      printf("test_7 run failed\n");
      exit(-1);
    }
  }
  printf("test_7 run pass!\n");
}

void test_8() {
  int A_keys[7] = {0, 1, 2, 2, 4, 6, 7};
  int A_vals[7] = {0, 0, 0, 0, 0, 0, 0};
  int B_keys[5] = {1, 1, 2, 5, 8};
  int B_vals[5] = {1, 1, 1, 1, 1};
  thrust::device_vector<int> d_keys_result(10);
  thrust::device_vector<int> d_vals_result(10);

  thrust::device_vector<int> d_A_keys(A_keys, A_keys + 7);
  thrust::device_vector<int> d_A_vals(A_vals, A_vals + 7);
  thrust::device_vector<int> d_B_keys(B_keys, B_keys + 5);
  thrust::device_vector<int> d_B_vals(B_vals, B_vals + 5);

  typedef thrust::device_vector<int>::iterator Iterator;

  thrust::pair<Iterator, Iterator> d_result;

  d_result = thrust::set_union_by_key(
      d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(),
      d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(),
      d_vals_result.begin(), Compare());

  int ref_keys[10] = {0, 1, 1, 2, 2, 4, 5, 6, 7, 8};
  int ref_vals[10] = {0, 0, 1, 0, 0, 0, 1, 0, 0, 1};
  for (int i = 0; i < 10; i++) {
    if (d_keys_result[i] != ref_keys[i]) {
      printf("test_8 run failed\n");
      exit(-1);
    }
  }

  for (int i = 0; i < 10; i++) {
    if (d_vals_result[i] != ref_vals[i]) {
      printf("test_8 run failed\n");
      exit(-1);
    }
  }

  printf("test_8 run pass\n");
}

void test_9() {
  int A_keys[7] = {0, 1, 2, 2, 4, 6, 7};
  int A_vals[7] = {0, 0, 0, 0, 0, 0, 0};
  int B_keys[5] = {1, 1, 2, 5, 8};
  int B_vals[5] = {1, 1, 1, 1, 1};
  thrust::host_vector<int> h_keys_result(10);
  thrust::host_vector<int> h_vals_result(10);

  thrust::host_vector<int> h_A_keys(A_keys, A_keys + 7);
  thrust::host_vector<int> h_A_vals(A_vals, A_vals + 7);
  thrust::host_vector<int> h_B_keys(B_keys, B_keys + 5);
  thrust::host_vector<int> h_B_vals(B_vals, B_vals + 5);

  typedef thrust::host_vector<int>::iterator Iterator;

  thrust::pair<Iterator, Iterator> h_result;

  h_result = thrust::set_union_by_key(
      thrust::host, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(),
      h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(),
      h_vals_result.begin());

  int ref_keys[10] = {0, 1, 1, 2, 2, 4, 5, 6, 7, 8};
  int ref_vals[10] = {
      0, 0, 1, 0, 0, 0, 1, 0, 0, 1,
  };
  for (int i = 0; i < 10; i++) {
    if (h_keys_result[i] != ref_keys[i]) {
      printf("test_9 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < 10; i++) {
    if (h_vals_result[i] != ref_vals[i]) {
      printf("test_9 run failed\n");
      exit(-1);
    }
  }
  printf("test_9 run pass!\n");
}

void test_10() {
  int A_keys[7] = {0, 1, 2, 2, 4, 6, 7};
  int A_vals[7] = {0, 0, 0, 0, 0, 0, 0};
  int B_keys[5] = {1, 1, 2, 5, 8};
  int B_vals[5] = {1, 1, 1, 1, 1};
  thrust::host_vector<int> h_keys_result(10);
  thrust::host_vector<int> h_vals_result(10);

  thrust::host_vector<int> h_A_keys(A_keys, A_keys + 7);
  thrust::host_vector<int> h_A_vals(A_vals, A_vals + 7);
  thrust::host_vector<int> h_B_keys(B_keys, B_keys + 5);
  thrust::host_vector<int> h_B_vals(B_vals, B_vals + 5);

  typedef thrust::host_vector<int>::iterator Iterator;

  thrust::pair<Iterator, Iterator> h_result;

  h_result = thrust::set_union_by_key(
      h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(),
      h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(),
      h_vals_result.begin());

  int ref_keys[10] = {0, 1, 1, 2, 2, 4, 5, 6, 7, 8};
  int ref_vals[10] = {
      0, 0, 1, 0, 0, 0, 1, 0, 0, 1,
  };
  for (int i = 0; i < 10; i++) {
    if (h_keys_result[i] != ref_keys[i]) {
      printf("test_10 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < 10; i++) {
    if (h_vals_result[i] != ref_vals[i]) {
      printf("test_10 run failed\n");
      exit(-1);
    }
  }
  printf("test_10 run pass\n");
}

void test_11() {
  int A_keys[7] = {0, 1, 2, 2, 4, 6, 7};
  int A_vals[7] = {0, 0, 0, 0, 0, 0, 0};
  int B_keys[5] = {1, 1, 2, 5, 8};
  int B_vals[5] = {1, 1, 1, 1, 1};
  thrust::host_vector<int> h_keys_result(10);
  thrust::host_vector<int> h_vals_result(10);

  thrust::host_vector<int> h_A_keys(A_keys, A_keys + 7);
  thrust::host_vector<int> h_A_vals(A_vals, A_vals + 7);
  thrust::host_vector<int> h_B_keys(B_keys, B_keys + 5);
  thrust::host_vector<int> h_B_vals(B_vals, B_vals + 5);

  typedef thrust::host_vector<int>::iterator Iterator;

  thrust::pair<Iterator, Iterator> h_result;

  h_result = thrust::set_union_by_key(
      thrust::host, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(),
      h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(),
      h_vals_result.begin(), Compare());

  int ref_keys[10] = {0, 1, 1, 2, 2, 4, 5, 6, 7, 8};
  int ref_vals[10] = {0, 0, 1, 0, 0, 0, 1, 0, 0, 1};
  for (int i = 0; i < 10; i++) {
    if (h_keys_result[i] != ref_keys[i]) {
      printf("test_11 run failed\n");
      exit(-1);
    }
  }

  for (int i = 0; i < 10; i++) {
    if (h_vals_result[i] != ref_vals[i]) {
      printf("test_11 run failed\n");
      exit(-1);
    }
  }
  printf("test_11 run pass!\n");
}

void test_12() {
  int A_keys[7] = {0, 1, 2, 2, 4, 6, 7};
  int A_vals[7] = {0, 0, 0, 0, 0, 0, 0};
  int B_keys[5] = {1, 1, 2, 5, 8};
  int B_vals[5] = {1, 1, 1, 1, 1};
  thrust::host_vector<int> h_keys_result(10);
  thrust::host_vector<int> h_vals_result(10);

  thrust::host_vector<int> h_A_keys(A_keys, A_keys + 7);
  thrust::host_vector<int> h_A_vals(A_vals, A_vals + 7);
  thrust::host_vector<int> h_B_keys(B_keys, B_keys + 5);
  thrust::host_vector<int> h_B_vals(B_vals, B_vals + 5);

  typedef thrust::host_vector<int>::iterator Iterator;

  thrust::pair<Iterator, Iterator> h_result;

  h_result = thrust::set_union_by_key(
      h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(),
      h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(),
      h_vals_result.begin(), Compare());

  int ref_keys[10] = {0, 1, 1, 2, 2, 4, 5, 6, 7, 8};
  int ref_vals[10] = {0, 0, 1, 0, 0, 0, 1, 0, 0, 1};
  for (int i = 0; i < 10; i++) {
    if (h_keys_result[i] != ref_keys[i]) {
      printf("test_12 run failed\n");
      exit(-1);
    }
  }

  for (int i = 0; i < 10; i++) {
    if (h_vals_result[i] != ref_vals[i]) {
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
