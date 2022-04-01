// ====------ onedpl_test_unique_by_key.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "oneapi/dpl/execution"
#include "oneapi/dpl/algorithm"
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
    return 0;
}

int test_passed(int failing_elems, std::string test_name) {
    if (failing_elems == 0) {
        std::cout << "PASS: " << test_name << std::endl;
        return 0;
    }
    return 1;
}

class UniqueByKey {};     // name for policy

int main() {

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";

    // #18 UNIQUE BY KEY TEST //

    {
        const int N = 10;
        cl::sycl::buffer<int, 1> keys_buf{ cl::sycl::range<1>(N) };
        cl::sycl::buffer<int, 1> values_buf{ cl::sycl::range<1>(N) };

        auto keys_it = oneapi::dpl::begin(keys_buf);
        auto values_it = oneapi::dpl::begin(values_buf);

        {
            auto keys = keys_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            auto values = values_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            keys[0] = 1; keys[1] = 1; keys[2] = 1; keys[3] = 4; keys[4] = 2; keys[5] = 2; keys[6] = 8;
            keys[7] = 5; keys[8] = 3; keys[9] = 3;
            values[0] = 'a'; values[1] = 'b'; values[2] = 'c'; values[3] = 'd'; values[4] = 'e';
            values[5] = 'f'; values[6] = 'g'; values[7] = 'h'; values[8] = 'i'; values[9] = 'j';
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<UniqueByKey>(oneapi::dpl::execution::dpcpp_default);

        // call algorithm:
        dpct::unique(new_policy, keys_it, keys_it + N, values_it);
        // keys is now   { 1,   4,   2,   8,   5,   3 }
        // values is now {'a', 'd', 'e', 'g', 'h', 'i'}

        {
            test_name = "unique test";
            auto values = values_it.get_buffer().template get_access<cl::sycl::access::mode::read>();

            num_failing += ASSERT_EQUAL(test_name, values[0], 'a');
            num_failing += ASSERT_EQUAL(test_name, values[1], 'd');
            num_failing += ASSERT_EQUAL(test_name, values[2], 'e');
            num_failing += ASSERT_EQUAL(test_name, values[3], 'g');
            num_failing += ASSERT_EQUAL(test_name, values[4], 'h');
            num_failing += ASSERT_EQUAL(test_name, values[5], 'i');

            failed_tests += test_passed(num_failing, test_name);
        }
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
