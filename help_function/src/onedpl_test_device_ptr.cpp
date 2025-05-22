// ====------ onedpl_test_device_ptr.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/iterator>

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <sycl/sycl.hpp>

#include <iostream>

template<typename String, typename _T1, typename _T2>
int ASSERT_EQUAL(String msg, _T1&& X, _T2&& Y) {
    if(X!=Y) {
        std::cout << "FAIL: " << msg << " - (" << X << "," << Y << ")" << std::endl;
        return 1;
    }
    else {
        std::cout << "PASS: " << msg << std::endl;
        return 0;
    }
}

int test_device_ptr_manipulation(void)
{
    int failing_tests = 0;
    typedef int T;

#ifdef DPCT_USM_LEVEL_NONE
    sycl::buffer<T, 1> data(sycl::range<1>(5));

    dpct::device_pointer<T> begin(data, 0);
    dpct::device_pointer<T> end(data, 5);
#else
    dpct::device_pointer<T> data(5 * sizeof(T));
    dpct::device_pointer<T> begin(data);
    dpct::device_pointer<T> end(data + 5);
#endif

    failing_tests += ASSERT_EQUAL("device_ptr test 1", end - begin, 5);

    begin++;
    begin--;

    failing_tests += ASSERT_EQUAL("device_ptr test 2", end - begin, 5);

    begin += 1;
    begin -= 1;

    failing_tests += ASSERT_EQUAL("device_ptr test 3", end - begin, 5);

    begin = begin + (int) 1;
    begin = begin - (int) 1;

    failing_tests += ASSERT_EQUAL("device_ptr test 4", end - begin, 5);

    begin = begin + (unsigned int) 1;
    begin = begin - (unsigned int) 1;

    failing_tests += ASSERT_EQUAL("device_ptr test 5", end - begin, 5);

    begin = begin + (size_t) 1;
    begin = begin - (size_t) 1;

    failing_tests += ASSERT_EQUAL("device_ptr test 6", end - begin, 5);

    begin = begin + (ptrdiff_t) 1;
    begin = begin - (ptrdiff_t) 1;

    failing_tests += ASSERT_EQUAL("device_ptr test 7", end - begin, 5);

    begin = begin + (dpct::device_pointer<T>::difference_type) 1;
    begin = begin - (dpct::device_pointer<T>::difference_type) 1;

    failing_tests += ASSERT_EQUAL("device_ptr test 8", end - begin, 5);

    dpct::device_iterator<T> iter(begin);
    *begin = 3;
    failing_tests += ASSERT_EQUAL("device_ptr test 9", *begin, 3);
    failing_tests += ASSERT_EQUAL("device_ptr test 10", *iter, 3);

    return failing_tests;
}

int test_device_ptr_iteration(void)
{
    int failing_tests = 0;
    typedef size_t T;

#ifdef DPCT_USM_LEVEL_NONE
    sycl::buffer<T, 1> data(sycl::range<1>(1024));

    dpct::device_pointer<T> begin(data, 0);
    dpct::device_pointer<T> end(data, 1024);
#else
    dpct::device_pointer<T> data(1024*sizeof(T));
    dpct::device_pointer<T> begin(data);
    dpct::device_pointer<T> end(data + 1024);
#endif
    auto policy = oneapi::dpl::execution::make_device_policy(dpct::get_default_queue());

    std::fill(policy, begin, end, 99);
    T result = oneapi::dpl::transform_reduce(policy, begin, end, static_cast<T>(0), std::plus<T>(), oneapi::dpl::identity());
    failing_tests += ASSERT_EQUAL("device_ptr in transform_reduce", result, 1024*99);
    return failing_tests;
}

int main() {

#ifndef DPCT_USM_LEVEL_NONE
    // If USM is used, then device_pointer must be device copyable to be passed directly into oneDPL
    static_assert(sycl::is_device_copyable_v<dpct::device_pointer<int>>);
    static_assert(sycl::is_device_copyable_v<dpct::device_pointer<void>>);

    static_assert(sycl::is_device_copyable_v<dpct::device_iterator<int>>);
    static_assert(sycl::is_device_copyable_v<dpct::device_iterator<void>>);
#endif

    int failed_tests = test_device_ptr_manipulation();
    failed_tests += test_device_ptr_iteration();

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
