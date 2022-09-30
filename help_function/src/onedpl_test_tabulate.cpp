// ====------ onedpl_test_tabulate.cpp---------- -*- C++ -* ----===////
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
#include <list>

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

class Sequence1 {};     // name for policy
class Sequence2 {};     // name for policy
class Sequence3 {};     // name for policy

int main() {
    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";

    // #16 TABULATE TEST //

    {
        // create buffer
        sycl::buffer<int64_t, 1> dst_buf{ sycl::range<1>(8) };

        auto dst_it = oneapi::dpl::begin(dst_buf);
        auto dst_end_it = oneapi::dpl::end(dst_buf);

        {
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::write>();
            dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0; dst[5] = 0; dst[6] = 0; dst[7] = 0;
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<Sequence1>(oneapi::dpl::execution::dpcpp_default);
        // call algorithm:
        dpct::for_each_index(new_policy, dst_it, dst_end_it, std::negate<int64_t>());

        {
            test_name = "tabulate test";
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::read>();
            num_failing += ASSERT_EQUAL(test_name, dst[0], 0);
            num_failing += ASSERT_EQUAL(test_name, dst[1], -1);
            num_failing += ASSERT_EQUAL(test_name, dst[2], -2);
            num_failing += ASSERT_EQUAL(test_name, dst[3], -3);
            num_failing += ASSERT_EQUAL(test_name, dst[4], -4);
            num_failing += ASSERT_EQUAL(test_name, dst[5], -5);
            num_failing += ASSERT_EQUAL(test_name, dst[6], -6);
            num_failing += ASSERT_EQUAL(test_name, dst[7], -7);

            failed_tests += test_passed(num_failing, test_name);
        }
    }

#if TEST_ALGO_STATIC_ASSERT
    {
        std::list<int64_t> l;

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<Sequence1>(oneapi::dpl::execution::dpcpp_default);

        // Algorithm call should fail due to non-random-access iterators being passed.
        dpct::for_each_index(new_policy, l.begin(), l.end(), std::negate<int64_t>());
    }
#endif

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
