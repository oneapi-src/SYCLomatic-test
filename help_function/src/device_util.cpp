// ====------ device_util.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

int main() {

  // test_feature:cpu_device()
  int _cpu_device = -1;
  std::vector<sycl::device> sycl_all_devs =
    sycl::device::get_devices(sycl::info::device_type::all);
  for (auto &dev : sycl_all_devs) {
    if (dev.is_cpu()) {
      _cpu_device = 1;
      break;
    }
  }
  if(_cpu_device == 1)
    dpct::cpu_device();

  // test_feature:get_default_queue()
  dpct::get_default_queue();

  // test_feature:get_current_device()
  dpct::get_current_device();

  // test_feature:get_device();
  dpct::get_device(0);

  // test_feature:get_default_context();
  dpct::get_default_context();

  // test_feature:exception_handler
  sycl::exception_list ExceptionList;
  dpct::exception_handler(ExceptionList);

  //test_feature:select_device(sycl::default_selector_v)
  dpct::select_device(sycl::default_selector_v);

  return 0;
}
