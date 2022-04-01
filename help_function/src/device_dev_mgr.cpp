// ====------ device_dev_mgr.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

int main() {

  // test_feature:instance()
  auto& Dmgr = dpct::dev_mgr::instance();

  // test_feature:current_device()
  Dmgr.current_device();

  // test_feature:cpu_device()
  int _cpu_device = -1;
  std::vector<cl::sycl::device> sycl_all_devs =
    cl::sycl::device::get_devices(cl::sycl::info::device_type::all);
  for (auto &dev : sycl_all_devs) {
    if (dev.is_cpu()) {
      _cpu_device = 1;
      break;
    }
  }
  if(_cpu_device == 1)
    Dmgr.cpu_device();

  // test_feature:get_device(0)
  Dmgr.get_device(0);

  // test_feature:current_device_id()
  Dmgr.current_device_id();

  // test_feature:select_device(0)
  Dmgr.select_device(0);

  // test_feature:device_count()
  Dmgr.device_count();

  return 0;
}
