// ====------ onedpl_test_run_length_encode.cpp---------- -*- C++ -* ----===////
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

#include <sycl/sycl.hpp>

#include <iostream>

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

  {
    // Test 1: Integral types with no edge runs.
    sycl::buffer<int64_t, 1> src_buf{sycl::range<1>(10)};
    sycl::buffer<int64_t, 1> dst_offsets_buf{sycl::range<1>(10)};
    sycl::buffer<int64_t, 1> dst_lengths_buf{sycl::range<1>(10)};
    sycl::buffer<int64_t, 1> dst_num_runs{sycl::range<1>(1)};
    auto src_it = oneapi::dpl::begin(src_buf);
    auto src_end_it = oneapi::dpl::end(src_buf);
    auto dst_offsets_it = oneapi::dpl::begin(dst_offsets_buf);
    auto dst_lengths_it = oneapi::dpl::begin(dst_lengths_buf);
    auto dst_num_runs_it = oneapi::dpl::begin(dst_num_runs);
    {
      auto src_acc = src_buf.get_host_access();
      src_acc[0] = 4, src_acc[1] = 2, src_acc[2] = 2, src_acc[3] = 2,
      src_acc[4] = 5, src_acc[5] = 6, src_acc[6] = 2, src_acc[7] = 8,
      src_acc[8] = 8, src_acc[9] = 4;
    }
    dpct::nontrivial_run_length_encode(oneapi::dpl::execution::dpcpp_default,
                                       src_it, src_end_it, dst_offsets_it,
                                       dst_lengths_it, dst_num_runs_it);
    {
      int offsets_check[2] = {1, 7};
      int lengths_check[2] = {3, 2};
      auto dst_offsets_acc = dst_offsets_buf.get_host_access();
      auto dst_lengths_acc = dst_lengths_buf.get_host_access();
      std::string test_name =
          "Call to dpct::nontrivial_run_length_encode with no edge runs";
      num_failing +=
          ASSERT_EQUAL(test_name, dst_num_runs.get_host_access()[0], 2);
      for (int i = 0; i < 2; ++i) {
        num_failing +=
            ASSERT_EQUAL(test_name, offsets_check[i], dst_offsets_acc[i]);
        num_failing +=
            ASSERT_EQUAL(test_name, lengths_check[i], dst_lengths_acc[i]);
      }
      failed_tests += test_passed(num_failing, test_name);
      num_failing = 0;
    }
  }
  {
    // Test 2: Integral types with edge runs.
    sycl::buffer<int64_t, 1> src_buf{sycl::range<1>(12)};
    sycl::buffer<int64_t, 1> dst_offsets_buf{sycl::range<1>(12)};
    sycl::buffer<int64_t, 1> dst_lengths_buf{sycl::range<1>(12)};
    sycl::buffer<int64_t, 1> dst_num_runs{sycl::range<1>(1)};
    auto src_it = oneapi::dpl::begin(src_buf);
    auto src_end_it = oneapi::dpl::end(src_buf);
    auto dst_offsets_it = oneapi::dpl::begin(dst_offsets_buf);
    auto dst_lengths_it = oneapi::dpl::begin(dst_lengths_buf);
    auto dst_num_runs_it = oneapi::dpl::begin(dst_num_runs);
    {
      auto src_acc = src_buf.get_host_access();
      src_acc[0] = 4, src_acc[1] = 4, src_acc[2] = 2, src_acc[3] = 2,
      src_acc[4] = 5, src_acc[5] = 6, src_acc[6] = 2, src_acc[7] = 8,
      src_acc[8] = 6, src_acc[9] = 3, src_acc[10] = 3, src_acc[11] = 3;
    }
    dpct::nontrivial_run_length_encode(oneapi::dpl::execution::dpcpp_default,
                                       src_it, src_end_it, dst_offsets_it,
                                       dst_lengths_it, dst_num_runs_it);
    {
      int offsets_check[3] = {0, 2, 9};
      int lengths_check[3] = {2, 2, 3};
      auto dst_offsets_acc = dst_offsets_buf.get_host_access();
      auto dst_lengths_acc = dst_lengths_buf.get_host_access();
      std::string test_name =
          "Call to dpct::nontrivial_run_length_encode with edge runs";
      num_failing +=
          ASSERT_EQUAL(test_name, dst_num_runs.get_host_access()[0], 3);
      for (int i = 0; i < 3; ++i) {
        num_failing +=
            ASSERT_EQUAL(test_name, offsets_check[i], dst_offsets_acc[i]);
        num_failing +=
            ASSERT_EQUAL(test_name, lengths_check[i], dst_lengths_acc[i]);
      }
      failed_tests += test_passed(num_failing, test_name);
      num_failing = 0;
    }
  }
  {
    // Test 3: Integral types with no runs at all
    sycl::buffer<int64_t, 1> src_buf{sycl::range<1>(5)};
    sycl::buffer<int64_t, 1> dst_offsets_buf{sycl::range<1>(5)};
    sycl::buffer<int64_t, 1> dst_lengths_buf{sycl::range<1>(5)};
    sycl::buffer<int64_t, 1> dst_num_runs{sycl::range<1>(1)};
    auto src_it = oneapi::dpl::begin(src_buf);
    auto src_end_it = oneapi::dpl::end(src_buf);
    auto dst_offsets_it = oneapi::dpl::begin(dst_offsets_buf);
    auto dst_lengths_it = oneapi::dpl::begin(dst_lengths_buf);
    auto dst_num_runs_it = oneapi::dpl::begin(dst_num_runs);
    {
      auto src_acc = src_buf.get_host_access();
      src_acc[0] = 0, src_acc[1] = 1, src_acc[2] = 2, src_acc[3] = 3,
      src_acc[4] = 4;
    }
    dpct::nontrivial_run_length_encode(oneapi::dpl::execution::dpcpp_default,
                                       src_it, src_end_it, dst_offsets_it,
                                       dst_lengths_it, dst_num_runs_it);
    {
      std::string test_name =
          "Call to dpct::nontrivial_run_length_encode with no runs at all";
      num_failing +=
          ASSERT_EQUAL(test_name, dst_num_runs.get_host_access()[0], 0);
      failed_tests += test_passed(num_failing, test_name);
      num_failing = 0;
    }
  }
  {
    // Test 4: Integral types with one long run
    sycl::buffer<int64_t, 1> src_buf{sycl::range<1>(5)};
    sycl::buffer<int64_t, 1> dst_offsets_buf{sycl::range<1>(5)};
    sycl::buffer<int64_t, 1> dst_lengths_buf{sycl::range<1>(5)};
    sycl::buffer<int64_t, 1> dst_num_runs{sycl::range<1>(1)};
    auto src_it = oneapi::dpl::begin(src_buf);
    auto src_end_it = oneapi::dpl::end(src_buf);
    auto dst_offsets_it = oneapi::dpl::begin(dst_offsets_buf);
    auto dst_lengths_it = oneapi::dpl::begin(dst_lengths_buf);
    auto dst_num_runs_it = oneapi::dpl::begin(dst_num_runs);
    {
      auto src_acc = src_buf.get_host_access();
      src_acc[0] = 2, src_acc[1] = 2, src_acc[2] = 2, src_acc[3] = 2,
      src_acc[4] = 2;
    }
    dpct::nontrivial_run_length_encode(oneapi::dpl::execution::dpcpp_default,
                                       src_it, src_end_it, dst_offsets_it,
                                       dst_lengths_it, dst_num_runs_it);
    {
      std::string test_name =
          "Call to dpct::nontrivial_run_length_encode with one single run";
      auto dst_offsets_acc = dst_offsets_buf.get_host_access();
      auto dst_lengths_acc = dst_lengths_buf.get_host_access();
      num_failing +=
          ASSERT_EQUAL(test_name, dst_num_runs.get_host_access()[0], 1);
      num_failing += ASSERT_EQUAL(test_name, dst_offsets_acc[0], 0);
      num_failing += ASSERT_EQUAL(test_name, dst_lengths_acc[0], 5);
      failed_tests += test_passed(num_failing, test_name);
      num_failing = 0;
    }
  }
  {
    // Test 5: Small integral type
    sycl::buffer<bool, 1> src_buf{sycl::range<1>(10)};
    sycl::buffer<std::size_t, 1> dst_offsets_buf{sycl::range<1>(10)};
    sycl::buffer<std::size_t, 1> dst_lengths_buf{sycl::range<1>(10)};
    sycl::buffer<int64_t, 1> dst_num_runs{sycl::range<1>(1)};
    auto src_it = oneapi::dpl::begin(src_buf);
    auto src_end_it = oneapi::dpl::end(src_buf);
    auto dst_offsets_it = oneapi::dpl::begin(dst_offsets_buf);
    auto dst_lengths_it = oneapi::dpl::begin(dst_lengths_buf);
    auto dst_num_runs_it = oneapi::dpl::begin(dst_num_runs);
    {
      auto src_acc = src_buf.get_host_access();
      src_acc[0] = 0, src_acc[1] = 1, src_acc[2] = 1, src_acc[3] = 0,
      src_acc[4] = 0, src_acc[5] = 1, src_acc[6] = 1, src_acc[7] = 0,
      src_acc[8] = 0, src_acc[9] = 1;
    }
    dpct::nontrivial_run_length_encode(oneapi::dpl::execution::dpcpp_default,
                                       src_it, src_end_it, dst_offsets_it,
                                       dst_lengths_it, dst_num_runs_it);
    {
      int offsets_check[4] = {1, 3, 5, 7};
      int lengths_check[4] = {2, 2, 2, 2};
      auto dst_offsets_acc = dst_offsets_buf.get_host_access();
      auto dst_lengths_acc = dst_lengths_buf.get_host_access();
      std::string test_name =
          "Call to dpct::nontrivial_run_length_encode with small input type";
      num_failing +=
          ASSERT_EQUAL(test_name, dst_num_runs.get_host_access()[0], 4);
      for (int i = 0; i < 4; ++i) {
        num_failing +=
            ASSERT_EQUAL(test_name, offsets_check[i], dst_offsets_acc[i]);
        num_failing +=
            ASSERT_EQUAL(test_name, lengths_check[i], dst_lengths_acc[i]);
      }
      failed_tests += test_passed(num_failing, test_name);
      num_failing = 0;
    }
  }

  std::cout << std::endl
            << failed_tests << " failing test(s) detected." << std::endl;
  if (failed_tests == 0) {
    return 0;
  }
  return 1;
}
