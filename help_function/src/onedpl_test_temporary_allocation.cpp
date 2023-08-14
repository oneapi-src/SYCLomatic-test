// ====------ onedpl_test_temporary_allocation.cpp---------- -*- C++ -*
// ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/execution"
#include "oneapi/dpl/iterator"

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <iostream>

#include <sycl/sycl.hpp>

#include <sys/resource.h>

template <typename String, typename _T1, typename _T2>
int ASSERT_EQUAL(String msg, _T1 &&X, _T2 &&Y) {
  if (X != Y) {
    std::cout << "FAIL: " << msg << " - (" << X << "," << Y << ")" << std::endl;
    return 1;
  }
  return 0;
}

int test_passed(int failing_elems, std::string test_name) {
  if (failing_elems == 0) {
    std::cout << "PASS: " << test_name << std::endl;
    return 0;
  }
  return 1;
}

int main() {
  // used to detect failures
  int failed_tests = 0;
  int num_failing = 0;

  // Test One, temporary allocation with device tag.
  {
    std::size_t num_elements = 10;
    dpct::device_sys_tag device_sys;
    auto [int64_ptr, elements_allocated] =
        dpct::get_temporary_allocation<int64_t>(device_sys, num_elements);
    int64_t num_data[10];
    std::iota(std::rbegin(num_data), std::rend(num_data), 1);

    dpct::get_default_queue().submit([&](sycl::handler &h) {
      h.memcpy(int64_ptr, &num_data, sizeof(int64_t) * num_elements);
    });
    dpct::get_default_queue().wait();

    dpct::get_default_queue().submit([&](sycl::handler &h) {
      h.memcpy(&num_data, int64_ptr, num_elements * sizeof(int64_t));
    });
    dpct::get_default_queue().wait();

    std::string test_name =
        "get and release temporary allocation - device memory with tag";
    failed_tests += ASSERT_EQUAL(test_name, elements_allocated, num_elements);
    for (std::size_t i = 0; i != num_elements; ++i)
      failed_tests += ASSERT_EQUAL(test_name, num_data[i],
                                   static_cast<int64_t>(num_elements - i));

    test_passed(failed_tests, test_name);
    dpct::release_temporary_allocation(device_sys, int64_ptr);
  }

  // Test Two, temporary allocation with host tag.
  {
    std::size_t num_elements = 10;
    dpct::host_sys_tag host_sys;
    auto [int64_ptr, elements_allocated] =
        dpct::get_temporary_allocation<int64_t>(host_sys, num_elements);
    int64_t num_data[10];
    std::iota(std::rbegin(num_data), std::rend(num_data), 1);

    ::std::memcpy(int64_ptr, &num_data, sizeof(int64_t) * num_elements);

    std::string test_name =
        "get and release temporary allocation - host memory with tag";
    failed_tests += ASSERT_EQUAL(test_name, elements_allocated, num_elements);
    for (std::size_t i = 0; i != num_elements; ++i)
      failed_tests += ASSERT_EQUAL(test_name, int64_ptr[i],
                                   static_cast<int64_t>(num_elements - i));

    test_passed(failed_tests, test_name);
    dpct::release_temporary_allocation(host_sys, int64_ptr);
  }

  // Test Three, temporary allocation with device policy.
  {
    std::size_t num_elements = 10;
    auto policy = oneapi::dpl::execution::dpcpp_default;
    auto [int64_ptr, elements_allocated] =
        dpct::get_temporary_allocation<int64_t>(policy, num_elements);
    int64_t num_data[10];
    std::iota(std::rbegin(num_data), std::rend(num_data), 1);

    dpct::get_default_queue().submit([&](sycl::handler &h) {
      h.memcpy(int64_ptr, &num_data, sizeof(int64_t) * num_elements);
    });
    dpct::get_default_queue().wait();

    dpct::get_default_queue().submit([&](sycl::handler &h) {
      h.memcpy(&num_data, int64_ptr, num_elements * sizeof(int64_t));
    });
    dpct::get_default_queue().wait();

    std::string test_name =
        "get and release temporary allocation - device memory with policy";
    failed_tests += ASSERT_EQUAL(test_name, elements_allocated, num_elements);
    for (std::size_t i = 0; i != num_elements; ++i)
      failed_tests += ASSERT_EQUAL(test_name, num_data[i],
                                   static_cast<int64_t>(num_elements - i));

    test_passed(failed_tests, test_name);
    dpct::release_temporary_allocation(policy, int64_ptr);
  }

  // Test Four, temporary allocation with host policy.
  {
    std::size_t num_elements = 10;
    auto policy = oneapi::dpl::execution::seq;
    auto [int64_ptr, elements_allocated] =
        dpct::get_temporary_allocation<int64_t>(policy, num_elements);
    int64_t num_data[10];
    std::iota(std::rbegin(num_data), std::rend(num_data), 1);

    ::std::memcpy(int64_ptr, &num_data, sizeof(int64_t) * num_elements);

    std::string test_name =
        "get and release temporary allocation - host memory with policy";
    failed_tests += ASSERT_EQUAL(test_name, elements_allocated, num_elements);
    for (std::size_t i = 0; i != num_elements; ++i)
      failed_tests += ASSERT_EQUAL(test_name, int64_ptr[i],
                                   static_cast<int64_t>(num_elements - i));

    test_passed(failed_tests, test_name);
    dpct::release_temporary_allocation(policy, int64_ptr);
  }

  std::cout << std::endl
            << failed_tests << " failing test(s) detected." << std::endl;
  if (failed_tests == 0) {
    return 0;
  }
  return 1;
}
