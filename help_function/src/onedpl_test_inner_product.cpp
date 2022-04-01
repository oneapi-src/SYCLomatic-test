// ====------ onedpl_test_inner_product.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "oneapi/dpl/execution"

#include "oneapi/dpl/numeric"
#include "oneapi/dpl/iterator"

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <CL/sycl.hpp>

#include <iostream>
#include <iomanip>

template<typename Iterator, typename T>
bool check_values(Iterator first, Iterator last, const T& val)
{
    return std::all_of(first, last,
        [&val](const T& x) { return x == val; });
}

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

class InnerProduct1 {};     // name for policy
class InnerProduct2 {};     // name for policy

int main() {

    // used to detect failures
    int failed_tests = 0;

    // #21 INNER PRODUCT TEST //

    {
        // create buffers
        cl::sycl::buffer<uint64_t, 1> src_buf{ cl::sycl::range<1>(8) };

        // declare iterators
        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);

        {
            auto src = src_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            // initialize data
            src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4; src[5] = 5; src[6] = 6; src[7] = 7;
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<InnerProduct1>(oneapi::dpl::execution::dpcpp_default);

        // call algorithm:
        uint64_t res = dpct::inner_product(new_policy, src_it, src_end_it, src_it, uint64_t(0));

        failed_tests += ASSERT_EQUAL("Innter Product Test 1", res, 140);
    }

    {
        // create buffers
        cl::sycl::buffer<uint64_t, 1> src_buf{ cl::sycl::range<1>(8) };

        // declare iterators
        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);

        // initialize data
        {
            auto src = src_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4; src[5] = 5; src[6] = 6; src[7] = 7;
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<InnerProduct2>(oneapi::dpl::execution::dpcpp_default);

        // call algorithm:
        uint64_t res = dpct::inner_product(new_policy, src_it, src_end_it, src_it, uint64_t(0),
                           std::plus<uint64_t>(), std::multiplies<uint64_t>());

        failed_tests += ASSERT_EQUAL("Innter Product Test 2", res, 140);
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}

