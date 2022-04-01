// ====------ onedpl_test_sequence.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
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

class Sequence1 {};     // name for policy
class Sequence2 {};     // name for policy
class Sequence3 {};     // name for policy

int main() {

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";

    // #9 SEQUENCE TEST //

    {
        // create buffer
        cl::sycl::buffer<uint64_t, 1> dst_buf{ cl::sycl::range<1>(8) };

        auto dst_it = oneapi::dpl::begin(dst_buf);
        auto dst_end_it = oneapi::dpl::end(dst_buf);

        {
            auto dst = dst_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0; dst[5] = 0; dst[6] = 0; dst[7] = 0;
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<Sequence1>(oneapi::dpl::execution::dpcpp_default);
        // call algorithm:
        dpct::iota(new_policy, dst_it, dst_end_it);

        test_name = "sequence test 1";
        auto dst = dst_it.get_buffer().template get_access<cl::sycl::access::mode::read>();
        num_failing += ASSERT_EQUAL(test_name, dst[0], 0);
        num_failing += ASSERT_EQUAL(test_name, dst[1], 1);
        num_failing += ASSERT_EQUAL(test_name, dst[2], 2);
        num_failing += ASSERT_EQUAL(test_name, dst[3], 3);
        num_failing += ASSERT_EQUAL(test_name, dst[4], 4);
        num_failing += ASSERT_EQUAL(test_name, dst[5], 5);
        num_failing += ASSERT_EQUAL(test_name, dst[6], 6);
        num_failing += ASSERT_EQUAL(test_name, dst[7], 7);

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }

    {
        // create buffer
        cl::sycl::buffer<uint64_t, 1> dst_buf{ cl::sycl::range<1>(8) };

        auto dst_it = oneapi::dpl::begin(dst_buf);
        auto dst_end_it = oneapi::dpl::end(dst_buf);

        {
            auto dst = dst_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0; dst[5] = 0; dst[6] = 0; dst[7] = 0;
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<Sequence2>(oneapi::dpl::execution::dpcpp_default);
        // call algorithm:
        dpct::iota(new_policy, dst_it, dst_end_it, uint64_t(1));

        test_name = "sequence test 2";
        auto dst = dst_it.get_buffer().template get_access<cl::sycl::access::mode::read>();
        num_failing += ASSERT_EQUAL(test_name, dst[0], 1);
        num_failing += ASSERT_EQUAL(test_name, dst[1], 2);
        num_failing += ASSERT_EQUAL(test_name, dst[2], 3);
        num_failing += ASSERT_EQUAL(test_name, dst[3], 4);
        num_failing += ASSERT_EQUAL(test_name, dst[4], 5);
        num_failing += ASSERT_EQUAL(test_name, dst[5], 6);
        num_failing += ASSERT_EQUAL(test_name, dst[6], 7);
        num_failing += ASSERT_EQUAL(test_name, dst[7], 8);

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }

    {
        // create buffer
        cl::sycl::buffer<uint64_t, 1> dst_buf{ cl::sycl::range<1>(8) };

        auto dst_it = oneapi::dpl::begin(dst_buf);
        auto dst_end_it = oneapi::dpl::end(dst_buf);

        {
            auto dst = dst_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0; dst[5] = 0; dst[6] = 0; dst[7] = 0;
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<Sequence3>(oneapi::dpl::execution::dpcpp_default);
        // call algorithm:
        dpct::iota(new_policy, dst_it, dst_end_it, uint64_t(1), uint64_t(2));

        test_name = "sequence test 3";
        auto dst = dst_it.get_buffer().template get_access<cl::sycl::access::mode::read>();
        num_failing += ASSERT_EQUAL(test_name, dst[0], 1);
        num_failing += ASSERT_EQUAL(test_name, dst[1], 3);
        num_failing += ASSERT_EQUAL(test_name, dst[2], 5);
        num_failing += ASSERT_EQUAL(test_name, dst[3], 7);
        num_failing += ASSERT_EQUAL(test_name, dst[4], 9);
        num_failing += ASSERT_EQUAL(test_name, dst[5], 11);
        num_failing += ASSERT_EQUAL(test_name, dst[6], 13);
        num_failing += ASSERT_EQUAL(test_name, dst[7], 15);

        failed_tests += test_passed(num_failing, test_name);
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
