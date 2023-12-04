// ====------ onedpl_test_sys_tag_utils.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <oneapi/dpl/execution>

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

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

int test_internal_policy_conversion(void) {
  int num_fails = 0;

  using seq_tag = dpct::internal::policy_or_tag_to_tag<decltype(
      oneapi::dpl::execution::seq)>::type;
  using unseq_tag = dpct::internal::policy_or_tag_to_tag_t<decltype(
      oneapi::dpl::execution::seq)>;
  using par_tag = dpct::internal::policy_or_tag_to_tag_t<decltype(
      oneapi::dpl::execution::par)>;
  using par_unseq_tag = dpct::internal::policy_or_tag_to_tag_t<decltype(
      oneapi::dpl::execution::par_unseq)>;
  using dev_tag = dpct::internal::policy_or_tag_to_tag_t<decltype(
      oneapi::dpl::execution::dpcpp_default)>;
  using reflect_host_tag =
      dpct::internal::policy_or_tag_to_tag_t<dpct::host_sys_tag>;
  using reflect_dev_tag =
      dpct::internal::policy_or_tag_to_tag_t<dpct::device_sys_tag>;

  num_fails += ASSERT_EQUAL("seq policy tag conversion",
                            std::is_same_v<seq_tag, dpct::host_sys_tag>, true);
  num_fails +=
      ASSERT_EQUAL("unseq policy tag conversion",
                   std::is_same_v<unseq_tag, dpct::host_sys_tag>, true);
  num_fails += ASSERT_EQUAL("par policy tag conversion",
                            std::is_same_v<par_tag, dpct::host_sys_tag>, true);
  num_fails +=
      ASSERT_EQUAL("par_unseq policy tag conversion",
                   std::is_same_v<par_unseq_tag, dpct::host_sys_tag>, true);
  num_fails +=
      ASSERT_EQUAL("dpcpp_default policy tag conversion",
                   std::is_same_v<dev_tag, dpct::device_sys_tag>, true);
  num_fails +=
      ASSERT_EQUAL("host tag reflection",
                   std::is_same_v<reflect_host_tag, dpct::host_sys_tag>, true);
  num_fails +=
      ASSERT_EQUAL("device tag reflection",
                   std::is_same_v<reflect_dev_tag, dpct::device_sys_tag>, true);

  return num_fails;
}

int test_internal_is_host_policy_or_tag(void) {
  int num_fails = 0;

  constexpr bool seq_is_host_tag =
      dpct::internal::is_host_policy_or_tag<decltype(
          oneapi::dpl::execution::seq)>::value;
  constexpr bool unseq_is_host_tag =
      dpct::internal::is_host_policy_or_tag_v<decltype(
          oneapi::dpl::execution::unseq)>;
  constexpr bool par_is_host_tag =
      dpct::internal::is_host_policy_or_tag_v<decltype(
          oneapi::dpl::execution::par)>;
  constexpr bool par_unseq_is_host_tag =
      dpct::internal::is_host_policy_or_tag_v<decltype(
          oneapi::dpl::execution::par)>;
  constexpr bool dev_is_host_tag =
      dpct::internal::is_host_policy_or_tag_v<decltype(
          oneapi::dpl::execution::dpcpp_default)>;
  constexpr bool host_tag_is_host_tag =
      dpct::internal::is_host_policy_or_tag_v<dpct::host_sys_tag>;
  constexpr bool dev_tag_is_host_tag =
      dpct::internal::is_host_policy_or_tag_v<dpct::device_sys_tag>;

  num_fails += ASSERT_EQUAL("seq policy is host", seq_is_host_tag, true);
  num_fails += ASSERT_EQUAL("unseq policy is host", unseq_is_host_tag, true);
  num_fails += ASSERT_EQUAL("par policy is host", par_is_host_tag, true);
  num_fails +=
      ASSERT_EQUAL("par_unseq policy is host", par_unseq_is_host_tag, true);
  num_fails +=
      ASSERT_EQUAL("dpcpp_default policy is host", dev_is_host_tag, false);
  num_fails += ASSERT_EQUAL("host tag is host", host_tag_is_host_tag, true);
  num_fails += ASSERT_EQUAL("device tag is host", dev_tag_is_host_tag, false);

  return num_fails;
}

int main() {
  int failed_tests = test_internal_policy_conversion();
  failed_tests += test_internal_is_host_policy_or_tag();

  std::cout << std::endl
            << failed_tests << " failing test(s) detected." << std::endl;
  if (failed_tests == 0) {
    return 0;
  }
  return 1;
}
