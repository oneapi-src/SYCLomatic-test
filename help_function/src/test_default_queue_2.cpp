// ====------ test_default_queue_2.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <iostream>

extern sycl::queue &get_queue_1(void);

sycl::queue &get_queue_2(void) { 
    auto v = dpct::malloc_device(32);
    std::cout << v.get() << std::endl;
    return dpct::get_default_queue();
}

int main() {
  if (&get_queue_1() == &get_queue_2()) {
    std::cout << "Test Passed\n"
              << "\n";
    return 0;
  } else {
    std::cout << "Test Failed\n"
              << "\n";
    return -1;
  }
}
