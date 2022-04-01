// ====------ blas_utils_get_transpose.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>
#include <stdio.h>

bool test_get_transpose() {
  if (dpct::get_transpose(0) != oneapi::mkl::transpose::nontrans)
    return false;
  if (dpct::get_transpose(1) != oneapi::mkl::transpose::trans)
    return false;
  if (dpct::get_transpose(2) != oneapi::mkl::transpose::conjtrans)
    return false;
  return true;
}

int main() {
  bool pass = true;
  if(!test_get_transpose()) {
    pass = false;
    printf("get_transpose fail\n");
  }
  return (pass ? 0 : 1);
}