// ====------ onedpl_test_max_element.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "oneapi/dpl/execution"
#include "oneapi/dpl/iterator"
#include "oneapi/dpl/algorithm"

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
    return 0;
}

int test_passed(int failing_elems, std::string test_name) {
    if (failing_elems == 0) {
        std::cout << "PASS: " << test_name << std::endl;
        return 0;
    }
    return 1;
}

struct squareFunctor {
    unsigned long operator() (unsigned long n) const {
        return (n*n);
    }
};

int main() {

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;

    {
        // Test One, regular calls to std::max_element
        sycl::buffer<int64_t,1> src_buf{ sycl::range<1>(12) };
        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);

        {
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::write>();
            // src = {2, 0, 11, 7, 8, 5, 4, 1, 3, 6, 10, 9}
            src[0] = 2; src[1] = 0; src[2] = 11; src[3] = 7; src[4] = 8;
            src[5] = 5; src[6] = 4; src[7] = 1; src[8] = 3; src[9] = 6;
            src[10] = 10; src[11] = 9;
        }

        // Call algorithm
        auto location_it = std::max_element(oneapi::dpl::execution::dpcpp_default, src_it, src_end_it);

        {
            std::string test_name = "Regular call to std::max_element";
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();
            size_t d = std::distance(src_it, location_it);
            num_failing += ASSERT_EQUAL(test_name, src[d], 11);
            int data_check[12] = {2, 0, 11, 7, 8, 5, 4, 1, 3, 6, 10, 9};

            for (int i = 0; i != 12; ++i) {
                num_failing += ASSERT_EQUAL(test_name, data_check[i], src[i]);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    {
        // Test Two, test call to std::max_element with transform_iterators as parameters
        sycl::buffer<int64_t,1> src_buf{ sycl::range<1>(10) };
        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);

        {
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::write>();
            // src = {1, 5, 4, 0, 6, 2, 3, 7, 9, 8}
            src[0] = 1; src[1] = 5; src[2] = 4; src[3] = 0; src[4] = 6;
            src[5] = 2; src[6] = 3; src[7] = 7; src[8] = 9; src[9] = 8;
        }

        // making tranform iterator that uses a squareFunctor
        auto trans_begin = oneapi::dpl::make_transform_iterator(src_it, squareFunctor());
        auto trans_end = trans_begin + 10;

        // Call algorithm
        auto location_it = std::max_element(oneapi::dpl::execution::dpcpp_default, trans_begin, trans_end);

        {
            std::string test_name = "std::max_element with trf_it";
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();
            size_t d = std::distance(trans_begin, location_it);

            // bug in transform_iterator dereference
            //num_failing += ASSERT_EQUAL(test_name, trans_begin[d], 81);
            num_failing += ASSERT_EQUAL(test_name, src[d], 9);

            int data_check[10] = {1, 5, 4, 0, 6, 2, 3, 7, 9, 8};
            for (int i = 0; i != 10; ++i) {
                num_failing += ASSERT_EQUAL(test_name, data_check[i], src[i]);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
