// ====------ lib_common_utils_mkl_get_version.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/lib_common_utils.hpp>

#include <iostream>

int main() {
  int major = 0, minor = 0, patch = 0;
  dpct::mkl_get_version(dpct::version_field::major, &major);
  dpct::mkl_get_version(dpct::version_field::update, &minor);
  dpct::mkl_get_version(dpct::version_field::patch, &patch);
 
  std::cout << major << ", " << minor << ", " << patch << std::endl;

  return 0;
}
