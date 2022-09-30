// ====------ util_DataType_test.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <cstdint>
#include <dpct/dpct.hpp>
#include <iostream>
#include <type_traits>

void util_DataType_test() {

  if (!std::is_same<dpct::DataType<float>::T2, float>::value)
    exit(-1);

  if (!std::is_same<dpct::DataType<sycl::float2>::T2,
                    std::complex<float>>::value)
    exit(-1);

  printf("util_DataType_test passed!\n");
}

int main() {

  util_DataType_test();

  return 0;
}