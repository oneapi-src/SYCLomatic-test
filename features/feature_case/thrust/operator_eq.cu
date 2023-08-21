// ====------ operator_eq.cu--------------- *- CUDA -*---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <complex>
#include <iostream>
#include <thrust/complex.h>
#include <thrust/optional.h>
#include <thrust/pair.h>

int main() {
  {
    thrust::pair<int, float> a(42, 3.14f);

    bool ret = a == a;
    if (!ret) {
      printf("test1 failed\n");
      exit(-1);
    }
  }

  {
    thrust::complex<float> a(3.14f, 3.14f);

    bool ret = a == a;
    if (!ret) {
      printf("test2 failed\n");
      exit(-1);
    }
  }

  {
    thrust::complex<float> a(3.14f, 3.14f);
    std::complex<float> b(3.14f, 3.14f);

    bool ret = a == b;
    if (!ret) {
      printf("test3 failed\n");
      exit(-1);
    }
  }

  {
    thrust::complex<float> a(3.14f, 3.14f);
    std::complex<float> b(3.14f, 3.14f);

    bool ret = b == a;
    if (!ret) {
      printf("test4 failed\n");
      exit(-1);
    }
  }

  {
    thrust::complex<float> a(3.14f, 0);
    float b = 3.14f;

    bool ret = b == a;
    if (!ret) {
      printf("test5 failed\n");
      exit(-1);
    }
  }

  {
    thrust::complex<float> a(3.14f, 0);
    float b = 3.14f;

    bool ret = a == b;
    if (!ret) {
      printf("test6 failed\n");
      exit(-1);
    }
  }

  {
    thrust::optional<int> a = thrust::nullopt;
    thrust::optional<int> b = thrust::nullopt;

    thrust::optional<int> c = 1;
    thrust::optional<int> d = 1;

    bool ret1 = a == b;
    bool ret2 = c == d;
    if (!ret1 || !ret2) {
      printf("test7 failed\n");
      exit(-1);
    }
  }

  {
    thrust::optional<int> a = thrust::nullopt;
    thrust::optional<int> b = 1;

    bool ret = a == b;
    if (ret) {
      printf("test8 failed\n");
      exit(-1);
    }
  }

  {
    thrust::optional<int> a = 1;
    int b = 1;

    bool ret = a == b;
    if (!ret) {
      printf("test9 failed\n");
      exit(-1);
    }
  }

  printf("test passed!\n");
  return 0;
}