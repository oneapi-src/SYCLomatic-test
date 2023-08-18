// ====------ onedpl_test_equal_range.cpp---------- -*- C++ -* ----===////
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

#include <iostream>
#include <iomanip>

#include <sycl/sycl.hpp>


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



class __greater {
public:
  template <typename _Xp, typename _Yp>
  bool operator()(_Xp &&__x, _Yp &&__y) const {
    return std::forward<_Xp>(__x) > std::forward<_Yp>(__y);
  }
};


int main() {

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;

    // equal_range

    {
        // create buffer
        sycl::buffer<int64_t, 1> src_buf{ sycl::range<1>(8) };

        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);

        {
            auto src = src_it.get_buffer().template get_access<sycl::access::mode::write>();

            src[0] = 0;
            src[1] = 2;
            src[2] = 2;
            src[3] = 2;
            src[4] = 4;
            src[5] = 6;
            src[6] = 6;
            src[7] = 8;

        }

        auto new_policy = oneapi::dpl::execution::make_device_policy(oneapi::dpl::execution::dpcpp_default);
        // call algorithm:
        {
            std::string test_name = "Call to dpct::equal_range greater than all";
            auto ret = dpct::equal_range(new_policy, src_it, src_end_it, 100);
            auto lower = ::std::get<0>(ret);
            auto upper = ::std::get<1>(ret);
            num_failing += ASSERT_EQUAL(test_name + " (lower)", lower - src_it, 8);
            num_failing += ASSERT_EQUAL(test_name + " (upper)", upper - src_it, 8);
            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
        {
            std::string test_name = "Call to dpct::equal_range less than all";
            auto ret = dpct::equal_range(new_policy, src_it, src_end_it, -1);
            auto lower = ::std::get<0>(ret);
            auto upper = ::std::get<1>(ret);
            num_failing += ASSERT_EQUAL(test_name + " (lower)", lower - src_it, 0);
            num_failing += ASSERT_EQUAL(test_name + " (upper)", upper - src_it, 0);
            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
        {
            std::string test_name = "Call to dpct::equal_range with none equal";
            auto ret = dpct::equal_range(new_policy, src_it, src_end_it, 5);
            auto lower = ::std::get<0>(ret);
            auto upper = ::std::get<1>(ret);
            num_failing += ASSERT_EQUAL(test_name + " (lower)", lower - src_it, 5);
            num_failing += ASSERT_EQUAL(test_name + " (upper)", upper - src_it, 5);
            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
        {
            std::string test_name = "Call to dpct::equal_range with one equal";
            auto ret = dpct::equal_range(new_policy, src_it, src_end_it, 4);
            auto lower = ::std::get<0>(ret);
            auto upper = ::std::get<1>(ret);
            num_failing += ASSERT_EQUAL(test_name + " (lower)", lower - src_it, 4);
            num_failing += ASSERT_EQUAL(test_name + " (upper)", upper - src_it, 5);
            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
        {
            std::string test_name = "Call to dpct::equal_range with multiple equal";
            auto ret = dpct::equal_range(new_policy, src_it, src_end_it, 2);
            auto lower = ::std::get<0>(ret);
            auto upper = ::std::get<1>(ret);
            num_failing += ASSERT_EQUAL(test_name + " (lower)", lower - src_it, 1);
            num_failing += ASSERT_EQUAL(test_name + " (upper)", upper - src_it, 4);
            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

    }

    // dcpt::equal_range with device_vector
    {
        dpct::device_vector<int> src_dv(8);

        auto src_it = src_dv.begin();
        auto src_end_it = src_dv.end();

        {
            src_dv[0] = 0;
            src_dv[1] = 2;
            src_dv[2] = 2;
            src_dv[3] = 2;
            src_dv[4] = 4;
            src_dv[5] = 6;
            src_dv[6] = 6;
            src_dv[7] = 8;
        }


        auto new_policy = oneapi::dpl::execution::make_device_policy(oneapi::dpl::execution::dpcpp_default);
        // call algorithm:
        {
            std::string test_name = "Call to dpct::equal_range greater than all";
            auto ret = dpct::equal_range(new_policy, src_it, src_end_it, 100);
            auto lower = ::std::get<0>(ret);
            auto upper = ::std::get<1>(ret);
            num_failing += ASSERT_EQUAL(test_name + " (lower)", lower - src_it, 8);
            num_failing += ASSERT_EQUAL(test_name + " (upper)", upper - src_it, 8);
            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
        {
            std::string test_name = "Call to dpct::equal_range less than all";
            auto ret = dpct::equal_range(new_policy, src_it, src_end_it, -1);
            auto lower = ::std::get<0>(ret);
            auto upper = ::std::get<1>(ret);
            num_failing += ASSERT_EQUAL(test_name + " (lower)", lower - src_it, 0);
            num_failing += ASSERT_EQUAL(test_name + " (upper)", upper - src_it, 0);
            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
        {
            std::string test_name = "Call to dpct::equal_range with none equal";
            auto ret = dpct::equal_range(new_policy, src_it, src_end_it, 5);
            auto lower = ::std::get<0>(ret);
            auto upper = ::std::get<1>(ret);
            num_failing += ASSERT_EQUAL(test_name + " (lower)", lower - src_it, 5);
            num_failing += ASSERT_EQUAL(test_name + " (upper)", upper - src_it, 5);
            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
        {
            std::string test_name = "Call to dpct::equal_range with one equal";
            auto ret = dpct::equal_range(new_policy, src_it, src_end_it, 4);
            auto lower = ::std::get<0>(ret);
            auto upper = ::std::get<1>(ret);
            num_failing += ASSERT_EQUAL(test_name + " (lower)", lower - src_it, 4);
            num_failing += ASSERT_EQUAL(test_name + " (upper)", upper - src_it, 5);
            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
        {
            std::string test_name = "Call to dpct::equal_range with multiple equal";
            auto ret = dpct::equal_range(new_policy, src_it, src_end_it, 2);
            auto lower = ::std::get<0>(ret);
            auto upper = ::std::get<1>(ret);
            num_failing += ASSERT_EQUAL(test_name + " (lower)", lower - src_it, 1);
            num_failing += ASSERT_EQUAL(test_name + " (upper)", upper - src_it, 4);
            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

    }

    // dcpt::equal_range with device_vector and custom comparator
    {
        dpct::device_vector<int> src_dv(8);

        auto src_it = src_dv.begin();
        auto src_end_it = src_dv.end();
        
        {
            src_dv[0] = 10;
            src_dv[1] = 8;
            src_dv[2] = 6;
            src_dv[3] = 5;
            src_dv[4] = 5;
            src_dv[5] = 5;
            src_dv[6] = 5;
            src_dv[7] = 2;
        }


        auto new_policy = oneapi::dpl::execution::make_device_policy(oneapi::dpl::execution::dpcpp_default);
        // call algorithm:
        {
            std::string test_name = "Call to dpct::equal_range greater than all";
            auto ret = dpct::equal_range(new_policy, src_it, src_end_it, 100, __greater());
            auto lower = ::std::get<0>(ret);
            auto upper = ::std::get<1>(ret);
            num_failing += ASSERT_EQUAL(test_name + " (lower)", lower - src_it, 0);
            num_failing += ASSERT_EQUAL(test_name + " (upper)", upper - src_it, 0);
            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
        {
            std::string test_name = "Call to dpct::equal_range less than all";
            auto ret = dpct::equal_range(new_policy, src_it, src_end_it, 1, __greater());
            auto lower = ::std::get<0>(ret);
            auto upper = ::std::get<1>(ret);
            num_failing += ASSERT_EQUAL(test_name + " (lower)", lower - src_it, 8);
            num_failing += ASSERT_EQUAL(test_name + " (upper)", upper - src_it, 8);
            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
        {
            std::string test_name = "Call to dpct::equal_range with none equal";
            auto ret = dpct::equal_range(new_policy, src_it, src_end_it, 3, __greater());
            auto lower = ::std::get<0>(ret);
            auto upper = ::std::get<1>(ret);
            num_failing += ASSERT_EQUAL(test_name + " (lower)", lower - src_it, 7);
            num_failing += ASSERT_EQUAL(test_name + " (upper)", upper - src_it, 7);
            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
        {
            std::string test_name = "Call to dpct::equal_range with one equal";
            auto ret = dpct::equal_range(new_policy, src_it, src_end_it, 6, __greater());
            auto lower = ::std::get<0>(ret);
            auto upper = ::std::get<1>(ret);
            num_failing += ASSERT_EQUAL(test_name + " (lower)", lower - src_it, 2);
            num_failing += ASSERT_EQUAL(test_name + " (upper)", upper - src_it, 3);
            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
        {
            std::string test_name = "Call to dpct::equal_range with multiple equal";
            auto ret = dpct::equal_range(new_policy, src_it, src_end_it, 5, __greater());
            auto lower = ::std::get<0>(ret);
            auto upper = ::std::get<1>(ret);
            num_failing += ASSERT_EQUAL(test_name + " (lower)", lower - src_it, 3);
            num_failing += ASSERT_EQUAL(test_name + " (upper)", upper - src_it, 7);
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
