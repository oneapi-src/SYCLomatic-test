// ====------ onedpl_test_remove_if.cpp---------- -*- C++ -* ----===////
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

struct Identity {
    template <typename T>
    T operator()(const T& x) const {
        return x;
    }
};

class RemoveIf {};     // name for policy
class RemoveCopyIf {}; // name for policy

int main() {

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";

    // #81 REMOVE IF TEST //

    {
        const int N = 6;
        cl::sycl::buffer<int, 1> a_buf{ cl::sycl::range<1>(N) };
        cl::sycl::buffer<int, 1> s_buf{ cl::sycl::range<1>(N) };
        //cl::sycl::buffer<int, 1> b_buf{ cl::sycl::range<1>(N) };

        auto A_it = oneapi::dpl::begin(a_buf);
        auto S_it = oneapi::dpl::begin(s_buf);

        {
            auto A = A_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            auto S = S_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            A[0] = 1; A[1] = 4; A[2] = 2; A[3] = 8; A[4] = 5; A[5] = 7;
            S[0] = 0; S[1] = 1; S[2] = 1; S[3] = 1; S[4] = 0; S[5] = 0;
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<RemoveIf>(oneapi::dpl::execution::dpcpp_default);
        // call algorithm:
        auto end = dpct::remove_if(new_policy, A_it, A_it + N, S_it, Identity{});

        {
            test_name = "remove_if test";
            auto A = A_it.get_buffer().template get_access<cl::sycl::access::mode::read>();
            num_failing += ASSERT_EQUAL(test_name, A[0], 1);
            num_failing += ASSERT_EQUAL(test_name, A[1], 5);
            num_failing += ASSERT_EQUAL(test_name, A[2], 7);
            num_failing += ASSERT_EQUAL(test_name, std::distance(A_it, end), 3);

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }

    // #82 REMOVE COPY IF TEST //

    {
        const int N = 6;
        cl::sycl::buffer<int, 1> a_buf{ cl::sycl::range<1>(N) };
        cl::sycl::buffer<int, 1> s_buf{ cl::sycl::range<1>(N) };
        cl::sycl::buffer<int, 1> b_buf{ cl::sycl::range<1>(N) };

        auto A_it = oneapi::dpl::begin(a_buf);
        auto S_it = oneapi::dpl::begin(s_buf);
        auto B_it = oneapi::dpl::begin(b_buf);

        {
            auto A = A_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            auto S = S_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            auto B = B_it.get_buffer().template get_access<cl::sycl::access::mode::write>();
            A[0] = 1; A[1] = 4; A[2] = 2; A[3] = 8; A[4] = 5; A[5] = 7;
            S[0] = 0; S[1] = 1; S[2] = 1; S[3] = 1; S[4] = 0; S[5] = 0;
            B[0] = 0; B[1] = 0; B[2] = 0; B[3] = 0; B[4] = 0; B[5] = 0;
        }

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<RemoveCopyIf>(oneapi::dpl::execution::dpcpp_default);
        // call algorithm:
        auto end = dpct::remove_copy_if(new_policy, A_it, A_it + N, S_it, B_it, Identity{});

        {
            test_name = "remove_copy_if test";
            auto B = B_it.get_buffer().template get_access<cl::sycl::access::mode::read>();
            num_failing += ASSERT_EQUAL(test_name, B[0], 1);
            num_failing += ASSERT_EQUAL(test_name, B[1], 5);
            num_failing += ASSERT_EQUAL(test_name, B[2], 7);
            num_failing += ASSERT_EQUAL(test_name, std::distance(B_it, end), 3);

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
