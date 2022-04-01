// ====------ thrust-list.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <list>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "report.h"

template<typename Iterator>
bool verify(Iterator begin, Iterator end, Iterator expected) {
  for (Iterator p = begin; p != end; p++, expected++) {
    if (*p != *expected)
      return false;
  }
  return true;
}

int main() {
  Report::start("Check std::list container can be used in device_vector constructor");

  const int N = 4;
  int initData[N] = {1, 2, 3, 4};
  std::list<int> list(initData, initData + N);
  thrust::device_vector<int> expected(initData, initData + N);
  thrust::device_vector<int> dv(list.begin(), list.end());
  Report::check("Verify correct device_vector initialization from std::list",
                verify(dv.begin(), dv.end(), expected.begin()), true);

  std::vector<int> vec(N);
  std::vector<int> vecExpected(initData, initData + N);
  thrust::copy(dv.begin(), dv.end(), vec.begin());
  Report::check("Verify correct copy back to std::vector",
                verify(vec.begin(), vec.end(), vecExpected.begin()), true);

  return Report::finish();
}
