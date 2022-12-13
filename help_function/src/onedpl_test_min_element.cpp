// ====------ onedpl_test_min_element.cpp---------- -*- C++ -* ----===////
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

template<typename Buffer> void first_buffer(Buffer &src, int start_index, int end_index, uint64_t value) {
    for (int i = start_index; i != end_index; ++i){
        src[i] = value - i;
    }
}

template<typename Buffer> void second_buffer(Buffer &src, int start_index, int end_index, uint64_t value) {
    for (int i = start_index; i != end_index; ++i){
        src[i] = value + i;
    }
}


struct squareFunctor {
    unsigned long operator() (unsigned long n) const {
        return (n*n);
    }
};


int main(){

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";

    {
        // Test One, test normal call to std::min_element
        sycl::buffer<int64_t,1> src_buf{ sycl::range<1>(30) };
        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);

        {
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::write>();
            first_buffer(src, 0, 30, 30); // src = {30, 29, 28, ..., 2, 1 }
        }

        // Call algorithm
        auto location_it = std::min_element(oneapi::dpl::execution::dpcpp_default, src_it, src_end_it);

        {
            test_name = "Regular call to std::min_element 1";
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();
            size_t d = std::distance(src_it, location_it);
            num_failing += ASSERT_EQUAL(test_name, src[d], 1);

            for (int i = 0; i != 30; ++i) {
                num_failing += ASSERT_EQUAL(test_name, src[i], 30 - i);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    {
        // Test Two, test normal call to std::min_element
        sycl::buffer<int64_t,1> src_buf{ sycl::range<1>(8) };
        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);

        {
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::write>();
            second_buffer(src, 0, 8, 0); // src = {0, 1, 2, ..., 7 }
        }

        // Call algorithm
        auto location_it = std::min_element(oneapi::dpl::execution::dpcpp_default, src_it, src_end_it);

        {
            test_name = "Regular call to std::min_element 2";
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();
            size_t d = std::distance(src_it, location_it);
            num_failing += ASSERT_EQUAL(test_name, src[d], 0);

            for (int i = 0; i != 8; ++i) {
                num_failing += ASSERT_EQUAL(test_name, src[i], i);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    {
        // Test Three, test normal call to std::min_element
        sycl::buffer<int64_t,1> src_buf{ sycl::range<1>(1) };
        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);

        {
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::write>();
            src[0] = 1; // src = {1}
        }

        // Call algorithm
        auto location_it = std::min_element(oneapi::dpl::execution::dpcpp_default, src_it, src_end_it);

        {
            test_name = "Regular call to std::min_element 3";
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();
            size_t d = std::distance(src_it, location_it);
            num_failing += ASSERT_EQUAL(test_name, src[d], 1); // smallest element check
            num_failing += ASSERT_EQUAL(test_name, src[0], 1); // unchanged buffer check
        }

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }

    {
        // Test Four, test call to std::min_element with transform_iterators as parameters
        sycl::buffer<int64_t,1> src_buf{ sycl::range<1>(10) };
        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);
        {
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::write>();
            // src = {5, 8, 2, 9, 7, 4, 6, 3, 0, 1}
            src[0] = 5; src[1] = 8; src[2] = 2; src[3] = 9; src[4] = 7;
            src[5] = 4; src[6] = 6; src[7] = 3; src[8] = 0; src[9] = 1;
        }

        // making tranform iterator that uses a squareFunctor
        auto trans_begin = oneapi::dpl::make_transform_iterator(src_it, squareFunctor());
        auto trans_end = trans_begin + 5;

        // Call algorithm
        auto location_it = std::min_element(oneapi::dpl::execution::dpcpp_default, trans_begin, trans_end);

        {
            test_name = "std::min_element with trf_it";
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();
            size_t d = std::distance(trans_begin, location_it);

            // bug in transform_iterator dereference.
            // num_failing += ASSERT_EQUAL(test_name, trans_begin[d], 4);
            num_failing += ASSERT_EQUAL(test_name, src[d], 2);

            int data_check[10] = {5, 8, 2, 9, 7, 4, 6, 3, 0, 1};
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
