// ====------ device_device_ext.cpp---------- -*- C++ -* ----===////
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

  auto& Device = dpct::get_current_device();

  // test_feature:is_native_atomic_supported()
  Device.is_native_atomic_supported();

  // test_feature:get_major_version()
  Device.get_major_version();

  // test_feature:get_minor_version()
  Device.get_minor_version();

  // test_feature:get_max_compute_units()
  Device.get_max_compute_units();

  // test_feature:get_max_clock_frequency()
  Device.get_max_clock_frequency();

  // test_feature:get_integrated()
  Device.get_integrated();

  dpct::device_info Info;
  // test_feature:get_device_info(Info)
  Device.get_device_info(Info);

  // test_feature:get_device_info()
  Info = Device.get_device_info();

  // test_feature:reset()
  Device.reset();

  // test_feature:default_queue()
  auto& Queue = Device.default_queue();

  // test_feature:queues_wait_and_throw()
  Device.queues_wait_and_throw();

  // test_feature:create_queue()
  auto QueuePtr = Device.create_queue();

  // test_feature:destroy_queue(QueuePtr)
  Device.destroy_queue(QueuePtr);

  QueuePtr = Device.create_queue();
  // test_feature:set_saved_queue(QueuePtr)
  Device.set_saved_queue(QueuePtr);

  // test_feature:get_saved_queue()
  QueuePtr = Device.get_saved_queue();

  // test_feature:get_context()
  auto Context = Device.get_context();

  return 0;
}
