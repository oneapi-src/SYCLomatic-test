// ====------ util_find_first_set.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

void find_first_set_test(int *test_result) {
  int a;
  long long int lla;
  int result;
  a = 0;
  result = dpct::ffs(a);
  if (result != 0) {
      *test_result = 1;
      return;
  }

  a = -2147483648;
  result = dpct::ffs(a);
  if (result != 32) {
      *test_result = 1;
      return;
  }

  a = 128;
  result = dpct::ffs(a);
  if (result != 8) {
      *test_result = 1;
      return;
  }

  a = 2147483647;
  result = dpct::ffs(a);
  if (result != 1) {
      *test_result = 1;
      return;
  }

  lla = -9223372036854775808ll;
  result = dpct::ffs(lla);
  if (result != 64) {
      *test_result = 1;
      return;
  }

  lla = -9223372036854775807ll;
  result = dpct::ffs(lla);
  if (result != 1) {
      *test_result = 1;
      return;
  }

  lla = -9223372034707292160ll;
  result = dpct::ffs(lla);
  if (result != 32) {
      *test_result = 1;
      return;
  }

  lla = 2147483648ll;
  result = dpct::ffs(lla);
  if (result != 32) {
      *test_result = 1;
      return;
  }

}

int main() {

    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    int *test_result, host_test_result = 0;

    test_result = sycl::malloc_shared<int>(sizeof(int), q_ct1);
    *test_result = 0;

    q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
            [=](sycl::nd_item<3> item_ct1) {
                find_first_set_test(test_result);
    });

    dev_ct1.queues_wait_and_throw();
    find_first_set_test(&host_test_result);
    if(*test_result != 0 || host_test_result != 0) {
        exit(-1);
    }
    printf("passed\n");


  return 0;
}
