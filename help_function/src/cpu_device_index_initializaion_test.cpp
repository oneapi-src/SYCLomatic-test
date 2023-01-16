// ====------ cpu_device_index_initializaion_test.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//



#include <sycl/sycl.hpp>
#include <iostream>
#include <dpct/dpct.hpp>

using namespace sycl;

int main() {
  std::cout << "CPU Device Name: " << dpct::dev_mgr::instance().cpu_device().get_info<info::device::name>() << "\n";
  printf("Test passed!\n");
  return 0;
}
