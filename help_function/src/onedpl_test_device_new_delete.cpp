// ====------ onedpl_test_device_new_delete.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <iostream>

struct integer_wrapper {
  integer_wrapper() : m_val(10) {}
  integer_wrapper(int val) : m_val(val) {}
  int m_val;
};

template <typename String, typename _T1, typename _T2>
int ASSERT_EQUAL(String msg, _T1 &&X, _T2 &&Y) {
  if (X != Y) {
    std::cout << "FAIL: " << msg << " - (" << X << "," << Y << ")" << std::endl;
    return 1;
  } else {
    std::cout << "PASS: " << msg << std::endl;
    return 0;
  }
}

int test_passed(int failing_elems, std::string test_name) {
  if (failing_elems == 0) {
    std::cout << "PASS: " << test_name << std::endl;
    return 0;
  }
  return 1;
}

int test_integer_wrapper(dpct::device_pointer<integer_wrapper> dev_array, int n,
                         integer_wrapper host_wrapper = integer_wrapper()) {
  int failures = 0;
  std::vector<integer_wrapper> h_array(dev_array, dev_array + n);
  for (int i = 0; i != n; ++i)
    failures += (host_wrapper.m_val != h_array[i].m_val);
  std::transform(dpl::execution::make_device_policy(dpct::get_default_queue()),
                 dev_array, dev_array + n, dev_array, [](integer_wrapper e) {
                   e.m_val = 32;
                   return e;
                 });
  host_wrapper.m_val = 32;
  h_array = std::vector<integer_wrapper>(dev_array, dev_array + n);
  for (int i = 0; i != n; ++i)
    failures += (host_wrapper.m_val != h_array[i].m_val);
  return failures;
}

int test_int64_t(dpct::device_pointer<int64_t> dev_array, int n) {
  int failures = 0;
  dpl::fill(dpl::execution::make_device_policy(dpct::get_default_queue()),
            dev_array, dev_array + n, 24);
  std::vector<int64_t> h_array(dev_array, dev_array + n);
  for (int i = 0; i != n; ++i)
    failures += (h_array[i] != 24);
  return failures;
}

int test_device_new_operator() {
  int failures = 0;
  int n = 100;
  // 1. Testing usage with trivial type
  {
    dpct::device_pointer<int64_t> dev_array = dpct::device_new<int64_t>(n);
    int local_fail = test_int64_t(dev_array, n);
    dpct::device_delete(dev_array, n);
    failures +=
        test_passed(local_fail, "Call to device_new (operator new) with "
                                "int64_t type and typed allocation");
  }
  // 2. Testing object construction - ensure default constructor is called
  {
    dpct::device_pointer<integer_wrapper> dev_array =
        dpct::device_new<integer_wrapper>(n);
    int local_fail = test_integer_wrapper(dev_array, n);
    dpct::device_delete(dev_array, n);
    failures += test_passed(
        local_fail, "Call to device_new (operator new) with custom struct");
  }
  return failures;
}
int test_device_new_placement() {
  int failures = 0;
  int n = 100;
  // 1. Testing with a typed device_pointer
  {
    dpct::device_pointer<int64_t> ptr = dpct::malloc_device<int64_t>(n);
    dpct::device_pointer<int64_t> dev_array = dpct::device_new<int64_t>(ptr, n);
    int local_fail = test_int64_t(dev_array, n);
    failures += test_passed(
        local_fail, "Call to device_new (placement new) and device_delete "
                    "with int64_t and typed allocation");
  }
  // 2. Testing with a void device_pointer
  {
    dpct::device_pointer<void> ptr =
        dpct::malloc_device(n * sizeof(integer_wrapper));
    dpct::device_pointer<integer_wrapper> dev_array =
        dpct::device_new<integer_wrapper>(ptr, n);
    int local_fail = test_integer_wrapper(dev_array, n);
    failures += test_passed(
        local_fail, "Call to device_new (placement new) and device_delete "
                    "with integer_wrapper and raw allocation");
  }
  // 3. Testing with a void device_pointer and custom initial value
  {
    dpct::device_pointer<void> ptr =
        dpct::malloc_device(n * sizeof(integer_wrapper));
    dpct::device_pointer<integer_wrapper> dev_array =
        dpct::device_new<integer_wrapper>(ptr, integer_wrapper(555), n);
    int local_fail = test_integer_wrapper(dev_array, n, integer_wrapper(555));
    failures += test_passed(
        local_fail,
        "Call to device_new (placement new) and device_delete "
        "with integer_wrapper, raw allocation, and non-default initial value");
  }
  return failures;
}

int main() {
  int failed_tests = test_device_new_operator();
  failed_tests += test_device_new_placement();

  std::cout << std::endl
            << failed_tests << " failing test(s) detected." << std::endl;
  if (failed_tests == 0) {
    return 0;
  }
  return 1;
}
