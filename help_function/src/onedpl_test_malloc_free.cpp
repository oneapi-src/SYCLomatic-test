// ====------ onedpl_test_malloc_free.cpp---------- -*- C++ -* ----===////
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

  // Test One, test normal calls for dpct::malloc and dpct::free by allocating a
  // certain number of bytes on device.
  {
    std::size_t num_elements = 5;
    dpct::device_sys_tag device_exec;
    dpct::tagged_pointer<dpct::device_sys_tag, void> double_ptr =
        dpct::malloc(device_exec, num_elements * sizeof(double));
    double num_data[] = {0.0, 1.0, 2.0, 3.0, 4.0};

    dpct::get_default_queue().submit([&](sycl::handler &h) {
      h.memcpy(double_ptr, &num_data, sizeof(double) * num_elements);
    });
    dpct::get_default_queue().wait();

    dpct::get_default_queue().submit([&](sycl::handler &h) {
      h.memcpy(&num_data, double_ptr, num_elements * sizeof(double));
    });
    dpct::get_default_queue().wait();

    std::string test_name =
        "malloc and free byte array allocation - device memory with tag";
    for (std::size_t i = 0; i != num_elements; ++i)
      failed_tests +=
          ASSERT_EQUAL(test_name, num_data[i], static_cast<double>(i));

    test_passed(failed_tests, test_name);

    dpct::free(device_exec, double_ptr);
  }

  // Test Two, test normal calls for dpct::malloc and dpct::free by allocating a
  // certain number of bytes on host.
  {
    std::size_t num_elements = 5;
    dpct::host_sys_tag host_exec;
    dpct::tagged_pointer<dpct::host_sys_tag, void> double_ptr =
        dpct::malloc(host_exec, num_elements * sizeof(double));
    double num_data[] = {0.0, 1.0, 2.0, 3.0, 4.0};

    ::std::memcpy(double_ptr, &num_data, num_elements * sizeof(double));

    std::string test_name =
        "malloc and free byte array allocation - host memory with tag";
    for (std::size_t i = 0; i != num_elements; ++i)
      failed_tests +=
          ASSERT_EQUAL(test_name, static_cast<double *>(double_ptr)[i],
                       static_cast<double>(i));
    test_passed(failed_tests, test_name);

    dpct::free(host_exec, double_ptr);
  }

  // Test three, dpct::malloc and dpct::free allocation for a certain number of
  // int64_t elements on device
  {
    std::size_t num_elements = 10;
    dpct::device_sys_tag device_exec;
    dpct::tagged_pointer<dpct::device_sys_tag, int64_t> int64_ptr =
        dpct::malloc<int64_t>(device_exec, num_elements);
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
        "malloc and free int64_t array allocation - device memory with tag";
    for (std::size_t i = 0; i != num_elements; ++i)
      failed_tests += ASSERT_EQUAL(test_name, num_data[i],
                                   static_cast<int64_t>(num_elements - i));

    test_passed(failed_tests, test_name);

    dpct::free(device_exec, int64_ptr);
  }

  // Test four, dpct::malloc and dpct::free allocation for a certain number of
  // int64_t elements on host
  {
    std::size_t num_elements = 10;
    dpct::host_sys_tag host_exec;
    dpct::tagged_pointer<dpct::host_sys_tag, int64_t> int64_ptr =
        dpct::malloc<int64_t>(host_exec, num_elements);
    int64_t num_data[10];
    std::iota(std::rbegin(num_data), std::rend(num_data), 1);

    ::std::memcpy(int64_ptr, &num_data, sizeof(int64_t) * num_elements);

    std::string test_name =
        "malloc and free int64_t array allocation - host memory with tag";
    for (std::size_t i = 0; i != num_elements; ++i)
      failed_tests += ASSERT_EQUAL(test_name, int64_ptr[i],
                                   static_cast<int64_t>(num_elements - i));

    test_passed(failed_tests, test_name);

    dpct::free(host_exec, int64_ptr);
  }

  // Test five, dpct::malloc and dpct::free allocation for a certain number of
  // int64_t elements on device. oneDPL device policy passed as location
  {
    std::size_t num_elements = 10;
    auto policy = oneapi::dpl::execution::dpcpp_default;
    dpct::tagged_pointer<dpct::device_sys_tag, int64_t> int64_ptr =
        dpct::malloc<int64_t>(policy, num_elements);
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
        "malloc and free int64_t array allocation - device memory with policy";
    for (std::size_t i = 0; i != num_elements; ++i)
      failed_tests += ASSERT_EQUAL(test_name, num_data[i],
                                   static_cast<int64_t>(num_elements - i));

    test_passed(failed_tests, test_name);

    dpct::free(policy, int64_ptr);
  }

  // Test six, dpct::malloc and dpct::free allocation for a certain number of
  // int64_t elements on device. oneDPL host policy passed as location
  {
    std::size_t num_elements = 10;
    auto policy = oneapi::dpl::execution::seq;
    dpct::tagged_pointer<dpct::host_sys_tag, int64_t> int64_ptr =
        dpct::malloc<int64_t>(policy, num_elements);
    int64_t num_data[10];
    ::std::iota(std::rbegin(num_data), std::rend(num_data), 1);

    ::std::memcpy(int64_ptr, &num_data, sizeof(int64_t) * num_elements);

    std::string test_name =
        "malloc and free int64_t array allocation - host memory with policy";
    for (std::size_t i = 0; i != num_elements; ++i)
      failed_tests += ASSERT_EQUAL(test_name, int64_ptr[i],
                                   static_cast<int64_t>(num_elements - i));

    test_passed(failed_tests, test_name);

    dpct::free(policy, int64_ptr);
  }

  // Test seven, ensure functionality of dpct::internal::malloc_base internal
  // utility.
  {
    std::size_t num_bytes = sizeof(int64_t) * 10;
    int64_t *ptr = static_cast<int64_t *>(
        dpct::internal::malloc_base(dpct::host_sys_tag{}, num_bytes));
    std::iota(ptr, ptr + 10, 0);
    std::string test_name = "internal::malloc_base utility with host tag";

    for (int i = 0; i < 10; ++i)
      failed_tests += ASSERT_EQUAL(test_name, ptr[i], i);
    std::free(ptr);
  }

  std::cout << std::endl
            << failed_tests << " failing test(s) detected." << std::endl;
  if (failed_tests == 0) {
    return 0;
  }
  return 1;
}
