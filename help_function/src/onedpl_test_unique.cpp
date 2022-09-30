// ====------ onedpl_test_unique.cpp---------- -*- C++ -* ----===////
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

#include <CL/sycl.hpp>

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

int main() {

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";

    {
        // Test One, test normal call with duplicates to std::unique
        sycl::buffer<int64_t,1> src_buf{ sycl::range<1>(16) };
        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);

        {
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::write>();
            // src = {2, 3, 3, 6, 7, 6, 1, 2, 0, 4, 4, 4, 4, 5, 7, 0}
            src[0] = 2; src[1] = 3; src[2] = 3; src[3] = 6; src[4] = 7; src[5] = 6;
            src[6] = 1; src[7] = 2; src[8] = 0; src[9] = 4; src[10] = 4; src[11] = 4;
            src[12] = 4; src[13] = 5; src[14] = 7; src[15] = 0;	
        }
        // Call algorithm
        auto location_it = std::unique(oneapi::dpl::execution::dpcpp_default, src_it, src_end_it); // returns location of new logical end
        {
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();
            size_t d = std::distance(src_it, location_it);
            test_name = "unique test with duplicates";

            num_failing += ASSERT_EQUAL(test_name, d, 12); // check that new size is correct
            int check[12] = {2, 3, 6, 7, 6, 1, 2, 0, 4, 5, 7, 0};

            for (int i = 0; i != 12; ++i) {
                num_failing += ASSERT_EQUAL(test_name, src[i], check[i]); // check that values are correct
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    {
        // Test Two, test normal call without duplicates to std::unique
        sycl::buffer<int64_t,1> src_buf{ sycl::range<1>(11) };
        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);

        {
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::write>();
            // src = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'x', 'y', 'z'}
            src[0] = 'a'; src[1] = 'b'; src[2] = 'c'; src[3] = 'd'; src[4] = 'e'; src[5] = 'f';
            src[6] = 'g'; src[7] = 'h'; src[8] = 'x'; src[9] = 'y'; src[10] = 'z';
        }

        // Call algorithm
        auto location_it = std::unique(oneapi::dpl::execution::dpcpp_default, src_it, src_end_it); // returns location of new logical end

        {
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::read>();
            size_t d = std::distance(src_it, location_it);
            test_name = "unique test without duplicates";

            num_failing += ASSERT_EQUAL(test_name, d, 11); // check that new size is correct
            int check[11] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'x', 'y', 'z'};

            for (int i = 0; i != 11; ++i) {
                num_failing += ASSERT_EQUAL(test_name, src[i], check[i]); // check that values are correct
            }

            failed_tests += test_passed(num_failing, test_name);
        }
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
