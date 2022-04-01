// ====------ onedpl_test_partition_copy.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "oneapi/dpl/execution"
#include "oneapi/dpl/algorithm"

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <CL/sycl.hpp>

#include <iostream>
#include <iomanip>

struct is_odd
{
    bool operator()(const int64_t &x) const {
        return x % 2;
    }
};

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

class PartitionCopy1 {};     // name for policy
class PartitionCopy2 {};     // name for policy

int main() {

    // #79 PARTITION COPY TEST //

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";

    {
        // create buffer
        cl::sycl::buffer<int64_t, 1> input_buf{ cl::sycl::range<1>(8) };
        cl::sycl::buffer<int64_t, 1> true_buf{ cl::sycl::range<1>(8) };
        cl::sycl::buffer<int64_t, 1> false_buf{ cl::sycl::range<1>(8) };

        auto inp_it = oneapi::dpl::begin(input_buf);
        auto inp_end_it = oneapi::dpl::end(input_buf);
        auto out_true_it = oneapi::dpl::begin(true_buf);
        auto out_false_it = oneapi::dpl::begin(false_buf);

        {
            auto inp = inp_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            auto out_true = out_true_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            auto out_false = out_false_it.get_buffer().template get_access<cl::sycl::access::mode::write>();

            for (int i = 0; i != 8; ++i) {
                inp[i] = i;
                out_true[i] = -1;
                out_false[i] = -1;
            }
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<PartitionCopy1>(oneapi::dpl::execution::dpcpp_default);

        // call algorithm:
        std::partition_copy(new_policy, inp_it, inp_end_it, out_true_it, out_false_it, is_odd());

        {
            test_name = "Regular call to std::partition_copy 1";
            auto out_true = out_true_it.get_buffer().template get_access<cl::sycl::access::mode::read>();
            auto out_false = out_false_it.get_buffer().template get_access<cl::sycl::access::mode::read>();

            num_failing += ASSERT_EQUAL(test_name, out_true[0], 1);
            num_failing += ASSERT_EQUAL(test_name, out_true[1], 3);
            num_failing += ASSERT_EQUAL(test_name, out_true[2], 5);
            num_failing += ASSERT_EQUAL(test_name, out_true[3], 7);
            num_failing += ASSERT_EQUAL(test_name, out_false[0], 0);
            num_failing += ASSERT_EQUAL(test_name, out_false[1], 2);
            num_failing += ASSERT_EQUAL(test_name, out_false[2], 4);
            num_failing += ASSERT_EQUAL(test_name, out_false[3], 6);
        }

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }

    {
        // create buffer
        cl::sycl::buffer<int64_t, 1> input_buf{ cl::sycl::range<1>(8) };
        cl::sycl::buffer<int64_t, 1> stencil_buf{ cl::sycl::range<1>(8) };
        cl::sycl::buffer<int64_t, 1> true_buf{ cl::sycl::range<1>(8) };
        cl::sycl::buffer<int64_t, 1> false_buf{ cl::sycl::range<1>(8) };

        auto inp_it = oneapi::dpl::begin(input_buf);
        auto inp_end_it = oneapi::dpl::end(input_buf);
        auto stn_it = oneapi::dpl::begin(stencil_buf);
        auto out_true_it = oneapi::dpl::begin(true_buf);
        auto out_false_it = oneapi::dpl::begin(false_buf);

        {
            auto inp = inp_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            auto stn = stn_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            auto out_true = out_true_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            auto out_false = out_false_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            for (int i = 0; i != 8; ++i) {
                inp[i] = i;
                stn[i] = i % 2;
                out_true[i] = -1;
                out_false[i] = -1;
            }
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<PartitionCopy2>(oneapi::dpl::execution::dpcpp_default);

        // call algorithm:
        dpct::partition_copy(new_policy, inp_it, inp_end_it, stn_it, out_true_it, out_false_it,
                               oneapi::dpl::identity());

        {
            test_name = "Regular call to std::partition_copy 2";
            auto out_true = out_true_it.get_buffer().template get_access<cl::sycl::access::mode::read>();
            auto out_false = out_false_it.get_buffer().template get_access<cl::sycl::access::mode::read>();

            num_failing += ASSERT_EQUAL(test_name, out_true[0], 1);
            num_failing += ASSERT_EQUAL(test_name, out_true[1], 3);
            num_failing += ASSERT_EQUAL(test_name, out_true[2], 5);
            num_failing += ASSERT_EQUAL(test_name, out_true[3], 7);
            num_failing += ASSERT_EQUAL(test_name, out_false[0], 0);
            num_failing += ASSERT_EQUAL(test_name, out_false[1], 2);
            num_failing += ASSERT_EQUAL(test_name, out_false[2], 4);
            num_failing += ASSERT_EQUAL(test_name, out_false[3], 6);
        }

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
