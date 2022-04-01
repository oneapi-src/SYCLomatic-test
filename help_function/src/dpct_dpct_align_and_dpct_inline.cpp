// ====------ dpct_dpct_align_and_dpct_inline.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

class __dpct_align__(8) T1 {
    unsigned int l, a;
};

class TestClass {
  TestClass();
  __dpct_inline__ void foo(){
    int a = 2;
  };
};

int main() {
  return 0;
}