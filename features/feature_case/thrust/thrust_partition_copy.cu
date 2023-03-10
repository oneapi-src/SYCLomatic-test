// ====------ thrust_partition_copy.cu------------- *- CUDA -*
// -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>

template <typename T> struct is_even {
  __host__ __device__ bool operator()(T x) const { return ((int)x % 2) == 0; }
};

void test_1() {

  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int result[10];
  const int N = sizeof(data) / sizeof(int);
  int *evens = result;
  int *odds = result + N / 2;
  thrust::partition_copy(thrust::host, data, data + N, evens, odds,
                         is_even<int>());

  int ref[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};

  for (int i = 0; i < N; i++) {
    if (result[i] != ref[i]) {
      std::cout << "test_1 run failed!\n";
      exit(-1);
    }
  }
  std::cout << "test_1 run passed!\n";
}

void test_2() {
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);
  thrust::host_vector<int> host_a(data, data + N);
  thrust::host_vector<int> host_evens(N / 2);
  thrust::host_vector<int> host_odds(N / 2);
  thrust::partition_copy(thrust::host, host_a.begin(), host_a.begin() + N,
                         host_evens.begin(), host_odds.begin(), is_even<int>());

  int ref_evens[] = {2, 4, 6, 8, 10};
  int ref_odds[] = {1, 3, 5, 7, 9};

  for (int i = 0; i < N / 2; i++) {
    if (host_evens[i] != ref_evens[i]) {
      std::cout << "test_2 run failed!\n";
      exit(-1);
    }
  }

  for (int i = 0; i < N / 2; i++) {
    if (host_odds[i] != ref_odds[i]) {
      std::cout << "test_2 run failed!"
                << "\n";
      exit(-1);
    }
  }

  std::cout << "test_2 run passed!\n";
}

void test_3() {
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);
  thrust::device_vector<int> device_a(data, data + N);
  thrust::device_vector<int> device_evens(N / 2);
  thrust::device_vector<int> device_odds(N / 2);
  thrust::partition_copy(thrust::device, device_a.begin(), device_a.begin() + N,
                         device_evens.begin(), device_odds.begin(),
                         is_even<int>());

  int ref_evens[] = {2, 4, 6, 8, 10};
  int ref_odds[] = {1, 3, 5, 7, 9};

  for (int i = 0; i < N / 2; i++) {
    if (device_evens[i] != ref_evens[i]) {
      std::cout << "test_3 run failed!\n";
      exit(-1);
    }
  }

  for (int i = 0; i < N / 2; i++) {
    if (device_odds[i] != ref_odds[i]) {
      std::cout << "test_3 run failed!"
                << "\n";
      exit(-1);
    }
  }

  std::cout << "test_3 run passed!\n";
}

void test_4() {

  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int result[10];
  const int N = sizeof(data) / sizeof(int);
  int *evens = result;
  int *odds = result + 5;
  thrust::partition_copy(data, data + N, evens, odds, is_even<int>());
  int ref[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};

  for (int i = 0; i < N; i++) {
    if (result[i] != ref[i]) {
      std::cout << "test_4 run failed!\n";
      exit(-1);
    }
  }
  std::cout << "test_4 run passed!\n";
}

void test_5() {
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);
  thrust::host_vector<int> host_a(data, data + N);
  thrust::host_vector<int> host_evens(N / 2);
  thrust::host_vector<int> host_odds(N / 2);
  thrust::partition_copy(host_a.begin(), host_a.begin() + N, host_evens.begin(),
                         host_odds.begin(), is_even<int>());

  int ref_evens[] = {2, 4, 6, 8, 10};
  int ref_odds[] = {1, 3, 5, 7, 9};

  for (int i = 0; i < N / 2; i++) {
    if (host_evens[i] != ref_evens[i]) {
      std::cout << "test_5 run failed!\n";
      exit(-1);
    }
  }

  for (int i = 0; i < N / 2; i++) {
    if (host_odds[i] != ref_odds[i]) {
      std::cout << "test_5 run failed!\n";
      exit(-1);
    }
  }

  std::cout << "test_5 run passed!\n";
}

void test_6() {
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);
  thrust::device_vector<int> device_a(data, data + N);
  thrust::device_vector<int> device_evens(N / 2);
  thrust::device_vector<int> device_odds(N / 2);
  thrust::partition_copy(device_a.begin(), device_a.begin() + N,
                         device_evens.begin(), device_odds.begin(),
                         is_even<int>());
  int ref_evens[] = {2, 4, 6, 8, 10};
  int ref_odds[] = {1, 3, 5, 7, 9};

  for (int i = 0; i < N / 2; i++) {
    if (device_evens[i] != ref_evens[i]) {
      std::cout << "test_6 run failed!\n";
      exit(-1);
    }
  }

  for (int i = 0; i < N / 2; i++) {
    if (device_odds[i] != ref_odds[i]) {
      std::cout << "test_6 run failed!\n";
      exit(-1);
    }
  }

  std::cout << "test_6 run passed!\n";
}

void test_7() {
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int S[] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  int result[10];
  const int N = sizeof(data) / sizeof(int);
  int *evens = result;
  int *odds = result + 5;
  thrust::partition_copy(thrust::host, data, data + N, S, evens, odds,
                         is_even<int>());

  int ref[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  for (int i = 0; i < N; i++) {
    if (result[i] != ref[i]) {
      std::cout << "test_7 run failed!\n";
    }
  }
  std::cout << "test_7 run passed!\n";
}

void test_8() {
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int S[] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  const int N = sizeof(data) / sizeof(int);
  thrust::host_vector<int> host_a(data, data + N);
  thrust::host_vector<int> host_evens(N / 2);
  thrust::host_vector<int> host_odds(N / 2);
  thrust::host_vector<int> host_S(S, S + N);
  thrust::partition_copy(thrust::host, host_a.begin(), host_a.begin() + N,
                         host_S.begin(), host_evens.begin(), host_odds.begin(),
                         is_even<int>());
  int ref_evens[] = {2, 4, 6, 8, 10};
  int ref_odds[] = {1, 3, 5, 7, 9};

  for (int i = 0; i < N / 2; i++) {
    if (host_evens[i] != ref_evens[i]) {
      std::cout << "test_8 run passed!\n";
    }
  }

  for (int i = 0; i < N / 2; i++) {
    if (host_odds[i] != ref_odds[i]) {
      std::cout << "test_8 run failed!\n";

      exit(-1);
    }
  }

  std::cout << "test_8 run passed!\n";
}

void test_9() {
  int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int S[] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  const int N = sizeof(A) / sizeof(int);
  thrust::device_vector<int> device_a(A, A + N);
  thrust::device_vector<int> device_evens(N / 2);
  thrust::device_vector<int> device_odds(N / 2);
  thrust::device_vector<int> device_S(S, S + N);
  thrust::partition_copy(thrust::device, device_a.begin(), device_a.begin() + N,
                         device_S.begin(), device_evens.begin(),
                         device_odds.begin(), is_even<int>());
  int ref_evens[] = {2, 4, 6, 8, 10};
  int ref_odds[] = {1, 3, 5, 7, 9};

  for (int i = 0; i < N / 2; i++) {
    if (device_evens[i] != ref_evens[i]) {
      std::cout << "test_9 run failed!\n";
      exit(-1);
    }
  }

  for (int i = 0; i < N / 2; i++) {
    if (device_odds[i] != ref_odds[i]) {
      std::cout << "test_9 run failed!\n";
      exit(-1);
    }
  }
  std::cout << "test_9 run passed!\n";
}

void test_10() {
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int S[] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};

  int result[10];
  const int N = sizeof(data) / sizeof(int);
  int *evens = result;
  int *odds = result + 5;
  thrust::partition_copy(data, data + N, S, evens, odds, is_even<int>());

  int ref[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  for (int i = 0; i < N; i++) {
    if (result[i] != ref[i]) {
      std::cout << "test_10 run failed!\n";
      exit(-1);
    }
  }
  std::cout << "test_10 run passed!\n";
}

void test_11() {
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int S[] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  const int N = sizeof(data) / sizeof(int);
  thrust::host_vector<int> host_a(data, data + N);
  thrust::host_vector<int> host_evens(N / 2);
  thrust::host_vector<int> host_odds(N / 2);
  thrust::host_vector<int> host_S(S, S + N);
  thrust::partition_copy(host_a.begin(), host_a.begin() + N, host_S.begin(),
                         host_evens.begin(), host_odds.begin(), is_even<int>());
  int ref_evens[] = {2, 4, 6, 8, 10};
  int ref_odds[] = {1, 3, 5, 7, 9};

  for (int i = 0; i < N / 2; i++) {
    if (host_evens[i] != ref_evens[i]) {
      std::cout << "test_11 run failed!\n";
      exit(-1);
    }
  }

  for (int i = 0; i < N / 2; i++) {
    if (host_odds[i] != ref_odds[i]) {
      std::cout << "test_11 run failed!\n";
      exit(-1);
    }
  }
  std::cout << "test_11 run passed!\n";
}

void test_12() {
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int S[] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  const int N = sizeof(data) / sizeof(int);
  thrust::device_vector<int> device_a(data, data + N);
  thrust::device_vector<int> device_evens(N / 2);
  thrust::device_vector<int> device_odds(N / 2);
  thrust::device_vector<int> device_S(S, S + N);
  thrust::partition_copy(device_a.begin(), device_a.begin() + N,
                         device_S.begin(), device_evens.begin(),
                         device_odds.begin(), is_even<int>());
  int ref_evens[] = {2, 4, 6, 8, 10};
  int ref_odds[] = {1, 3, 5, 7, 9};

  for (int i = 0; i < N / 2; i++) {
    if (device_evens[i] != ref_evens[i]) {
      std::cout << "test_12 run failed!"
                << "\n";
      exit(-1);
    }
  }

  for (int i = 0; i < N / 2; i++) {
    if (device_odds[i] != ref_odds[i]) {
      std::cout << "test_12 run failed!"
                << "\n";
      exit(-1);
    }
  }

  std::cout << "test_12 run passed!\n";
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
