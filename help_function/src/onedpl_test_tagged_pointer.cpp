// ====------ onedpl_test_tagged_pointer.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// Adapted from onedpl_test_device_ptr.cpp

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/numeric>

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <sycl/sycl.hpp>

#include <iostream>

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

template <typename SystemTag> int test_tagged_pointer_manipulation(void) {
  int failing_tests = 0;
  constexpr ::std::size_t n = 20;
  SystemTag system;
  ::std::string sys = ::std::is_same_v<dpct::host_sys_tag, SystemTag>
                          ? "dpct::host_sys_tag"
                          : "dpct::device_sys_tag";
  std::string int_ptr_name = "dpct::tagged_pointer<" + sys + ", int>";
  std::string void_ptr_name = "dpct::tagged_pointer<" + sys + ", void>";

  dpct::tagged_pointer<SystemTag, void> void_ptr_beg =
      dpct::malloc(system, sizeof(int) * n);
  dpct::tagged_pointer<SystemTag, int> int_ptr_beg =
      static_cast<dpct::tagged_pointer<SystemTag, int>>(void_ptr_beg);

  dpct::tagged_pointer<SystemTag, void> void_ptr_beg2 =
      static_cast<dpct::tagged_pointer<SystemTag, void>>(int_ptr_beg);
  failing_tests += ASSERT_EQUAL(void_ptr_name + " conversion operator",
                                void_ptr_beg == void_ptr_beg2, true);

  dpct::tagged_pointer<SystemTag, int> int_ptr_end = int_ptr_beg + n;
  failing_tests += ASSERT_EQUAL(
      int_ptr_name + " add operator",
      static_cast<int *>(int_ptr_end) - static_cast<int *>(int_ptr_beg), n);

  dpct::tagged_pointer<SystemTag, int> expect_beg = int_ptr_end - n;
  failing_tests += ASSERT_EQUAL(int_ptr_name + " subtract operator",
                                int_ptr_beg == expect_beg, true);

  failing_tests += ASSERT_EQUAL(int_ptr_name + " difference operator",
                                int_ptr_end - int_ptr_beg, n);

  expect_beg++;
  expect_beg--;
  failing_tests +=
      ASSERT_EQUAL(int_ptr_name + " postfix increment and decrement",
                   int_ptr_beg == expect_beg, true);

  ++expect_beg;
  --expect_beg;
  failing_tests +=
      ASSERT_EQUAL(int_ptr_name + " prefix increment and decrement",
                   int_ptr_beg == expect_beg, true);

  expect_beg += 2;
  expect_beg -= 2;
  failing_tests +=
      ASSERT_EQUAL(int_ptr_name + " addition and subtraction assignment",
                   int_ptr_beg == expect_beg, true);

  // Test conversion to base pointer
  int *int_ptr_beg_raw = int_ptr_beg;
  int *int_ptr_end_raw = int_ptr_end;
  failing_tests +=
      ASSERT_EQUAL(int_ptr_name + " tagged_pointer<int> conversion to int*",
                   int_ptr_end_raw - int_ptr_beg_raw, n);

  dpct::free(system, void_ptr_beg);
  return failing_tests;
}

template <typename SystemTag> int test_tagged_pointer_iteration(void) {
  constexpr ::std::size_t n = 1024;
  int return_fail_code = 0;
  SystemTag system;
  bool is_host = ::std::is_same_v<dpct::host_sys_tag, SystemTag>;
  ::std::string sys = is_host ? "dpct::host_sys_tag" : "dpct::device_sys_tag";
  std::string int_ptr_name = "dpct::tagged_pointer<" + sys + ", int>";

  auto policy = oneapi::dpl::execution::dpcpp_default;
  dpct::tagged_pointer<SystemTag, int> ptr_beg = dpct::malloc<int>(system, n);
  dpct::tagged_pointer<SystemTag, int> ptr_end = ptr_beg + n;

  std::fill(policy, ptr_beg, ptr_end, 99);
  int result = oneapi::dpl::reduce(policy, ptr_beg, ptr_beg + n);
  return_fail_code +=
      ASSERT_EQUAL(int_ptr_name + " reduce algorithm test", result, n * 99);
  dpct::free(system, ptr_beg);
  return return_fail_code;
}

int main() {
  int failed_tests = test_tagged_pointer_manipulation<dpct::host_sys_tag>();
  failed_tests += test_tagged_pointer_manipulation<dpct::device_sys_tag>();

  failed_tests += test_tagged_pointer_iteration<dpct::host_sys_tag>();
  failed_tests += test_tagged_pointer_iteration<dpct::device_sys_tag>();

  std::cout << std::endl
            << failed_tests << " failing test(s) detected." << std::endl;
  if (failed_tests == 0) {
    return 0;
  }
  return 1;
}
