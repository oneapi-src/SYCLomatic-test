// ====------ max.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

int foo(int i) {
  return sycl::max(0, i);
}

int main() {
  if (foo(23) == 23) {
    std::cout <<"Max value is 23" << std::endl;
    return 0;
  }
  std::cout <<"the api execution failed" << std::endl;
  return -1;
}

