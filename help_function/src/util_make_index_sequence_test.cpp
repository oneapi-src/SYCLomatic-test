// ====------ util_make_index_sequence_test.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

const int ref_range[3] = {1, 2, 3};

template <int... DimIdx>
sycl::range<sizeof...(DimIdx)>
get_range(dpct::integer_sequence<DimIdx...>) {
  return sycl::range<sizeof...(DimIdx)>(ref_range[DimIdx]...);
}

void make_index_sequence_test() {
  const int dimensions = 3;
  auto index = dpct::make_index_sequence<dimensions>();

  auto value = get_range(index);

  for (int i = 0; i < dimensions; i++) {
    if (value[i] != ref_range[i]) {
      printf("make_index_sequence_test failed\n");
      exit(-1);
    }
  }

  printf("make_index_sequence_test passed!\n");
}

int main() {

  make_index_sequence_test();

  return 0;
}