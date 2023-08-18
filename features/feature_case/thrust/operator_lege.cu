// ====------ operator_lege.cu------------- *- CUDA -*---------------------===//
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
    thrust::pair<int, float> x(42, 3.14f);
    thrust::pair<int, float> y(43, 3.24f);

    bool ret = x < y;
    if (!ret) {
      printf("test1 failed\n");
      exit(-1);
    }
  }

  {
    thrust::pair<int, float> x(42, 3.14f);
    thrust::pair<int, float> y(43, 3.24f);

    bool ret = x <= y;
    if (!ret) {
      printf("test2 failed\n");
      exit(-1);
    }
  }

  {
    thrust::pair<int, float> x(42, 3.14f);
    thrust::pair<int, float> y(43, 3.24f);

    bool ret = x > y;
    if (ret) {
      printf("test3 failed\n");
      exit(-1);
    }
  }

  {
    thrust::pair<int, float> x(42, 3.14f);
    thrust::pair<int, float> y(43, 3.24f);

    bool ret = x >= y;
    if (ret) {
      printf("test4 failed\n");
      exit(-1);
    }
  }
  printf("test passed!\n");
  return 0;
}