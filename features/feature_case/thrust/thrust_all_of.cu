// ====------ thrust_all_of.cu--------------- *- CUDA -*------------------===//
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
#include <thrust/logical.h>

void test_1() {
  bool A[3] = {true, true, false};
  bool result =
      thrust::all_of(thrust::host, A, A + 2, thrust::identity<bool>());
  if (!result) {
    printf("test_1 1 run failed\n");
    exit(-1);
  }

  result = thrust::all_of(thrust::host, A, A + 3, thrust::identity<bool>());

  if (result) {
    printf("test_1 2 run failed\n");
    exit(-1);
  }

  result = thrust::all_of(thrust::host, A, A, thrust::identity<bool>());

  if (!result) {
    std::cout << " " << result << "\n";
    printf("test_1 3 run failed\n");
    exit(-1);
  }

  printf("test_1 run passed!\n");
}

void test_2() {
  bool A[3] = {true, true, false};
  bool result = thrust::all_of(A, A + 2, thrust::identity<bool>());
  if (!result) {
    printf("test_2 1 run failed\n");
    exit(-1);
  }

  result = thrust::all_of(A, A + 3, thrust::identity<bool>());

  if (result) {
    printf("test_2 2 run failed\n");
    exit(-1);
  }

  result = thrust::all_of(A, A, thrust::identity<bool>());

  if (!result) {
    std::cout << " " << result << "\n";
    printf("test_2 3 run failed\n");
    exit(-1);
  }

  printf("test_2 run passed!\n");
}

void test_3() {
  bool A[3] = {true, true, false};

  thrust::host_vector<bool> h_A(A, A + 3);

  bool result = thrust::all_of(thrust::host, h_A.begin(), h_A.begin() + 2,
                               thrust::identity<bool>());
  if (!result) {
    printf("test_3 1 run failed\n");
    exit(-1);
  }

  result = thrust::all_of(thrust::host, h_A.begin(), h_A.begin() + 3,
                          thrust::identity<bool>());

  if (result) {
    printf("test_3 2 run failed\n");
    exit(-1);
  }

  result = thrust::all_of(thrust::host, h_A.begin(), h_A.begin(),
                          thrust::identity<bool>());

  if (!result) {
    std::cout << " " << result << "\n";
    printf("test_3 3 run failed\n");
    exit(-1);
  }

  printf("test_3 run passed!\n");
}

void test_4() {
  bool A[3] = {true, true, false};

  thrust::host_vector<bool> h_A(A, A + 3);

  bool result =
      thrust::all_of(h_A.begin(), h_A.begin() + 2, thrust::identity<bool>());
  if (!result) {
    printf("test_4 1 run failed\n");
    exit(-1);
  }

  result =
      thrust::all_of(h_A.begin(), h_A.begin() + 3, thrust::identity<bool>());

  if (result) {
    printf("test_4 2 run failed\n");
    exit(-1);
  }

  result = thrust::all_of(h_A.begin(), h_A.begin(), thrust::identity<bool>());

  if (!result) {
    std::cout << " " << result << "\n";
    printf("test_4 3 run failed\n");
    exit(-1);
  }

  printf("test_4 run passed!\n");
}

void test_5() {
  bool A[3] = {true, true, false};

  thrust::device_vector<bool> d_A(A, A + 3);

  bool result = thrust::all_of(thrust::device, d_A.begin(), d_A.begin() + 2,
                               thrust::identity<bool>());
  if (!result) {
    printf("test_5 1 run failed\n");
    exit(-1);
  }

  result = thrust::all_of(thrust::device, d_A.begin(), d_A.begin() + 3,
                          thrust::identity<bool>());

  if (result) {
    printf("test_5 2 run failed\n");
    exit(-1);
  }

  result = thrust::all_of(thrust::device, d_A.begin(), d_A.begin(),
                          thrust::identity<bool>());

  if (!result) {
    std::cout << " " << result << "\n";
    printf("test_5 3 run failed\n");
    exit(-1);
  }

  printf("test_5 run passed!\n");
}

void test_6() {
  bool A[3] = {true, true, false};

  thrust::device_vector<bool> d_A(A, A + 3);

  bool result =
      thrust::all_of(d_A.begin(), d_A.begin() + 2, thrust::identity<bool>());
  if (!result) {
    printf("test_6 1 run failed\n");
    exit(-1);
  }

  result =
      thrust::all_of(d_A.begin(), d_A.begin() + 3, thrust::identity<bool>());

  if (result) {
    printf("test_6 2 run failed\n");
    exit(-1);
  }

  result = thrust::all_of(d_A.begin(), d_A.begin(), thrust::identity<bool>());

  if (!result) {
    std::cout << " " << result << "\n";
    printf("test_6 3 run failed\n");
    exit(-1);
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