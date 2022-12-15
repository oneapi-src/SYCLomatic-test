// ====------ util_cast_double_from_to_int.cpp ----------- -*- C++ -* -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//


#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>

void k1(int *i, double d) {
  i[0] = dpct::cast_double_to_int(d, false);
  i[1] = dpct::cast_double_to_int(d);
}

bool test_cast_double_to_int() {
  double d;
  int i[2];
  i[0] = 127;
  i[1] = 255;
  memcpy(&d, i, sizeof(double));
  int *i_d;
  i_d = sycl::malloc_device<int>(2, dpct::get_default_queue());
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) {
        k1(i_d, d);
      });
  dpct::get_current_device().queues_wait_and_throw();
  int i_h[2];
  dpct::get_default_queue().memcpy(i_h, i_d, 2 * sizeof(int)).wait();
  sycl::free(i_d, dpct::get_default_queue());
  if ((i_h[0] == 127) && (i_h[1] == 255)) {
    std::cout << "test_cast_double_to_int pass" << std::endl;
    return true;
  }
  std::cout << "expect i_h[0]:127, i_h[1]:255" << std::endl;
  std::cout << "test result i_h[0]:" << i_h[0] << ", i_h[1]:" << i_h[1] << std::endl;
  std::cout << "test_cast_double_to_int fail" << std::endl;
  return false;
}

void k2(double *d) {
  int i1 = 127;
  int i2 = 255;
  *d = dpct::cast_ints_to_double(i1, i2);
}

bool test_cast_ints_to_double() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  double *d_d;
  d_d = sycl::malloc_device<double>(1, q_ct1);
  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) {
        k2(d_d);
      });
  dev_ct1.queues_wait_and_throw();
  double d_h;
  q_ct1.memcpy(&d_h, d_d, sizeof(double)).wait();
  sycl::free(d_d, q_ct1);
  std::int64_t int64 = *(reinterpret_cast<std::int64_t*>(&d_h));
  if (int64 == 545460846847) {
    std::cout << "test_cast_ints_to_double pass" << std::endl;
    return true;
  }
  std::cout << "expect int64:545460846847" << std::endl;
  std::cout << "test result int64:" << int64 << std::endl;
  std::cout << "test_cast_ints_to_double fail" << std::endl;
  return false;
}

int main() {
  bool pass = true;
  pass = pass && test_cast_double_to_int();
  pass = pass && test_cast_ints_to_double();
  if (pass)
    return 0;
  return 1;
}
